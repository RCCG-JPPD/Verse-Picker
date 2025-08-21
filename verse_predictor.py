import os, sys, json, time, queue, threading, re, argparse
from datetime import datetime, timezone
from pathlib import Path
import html
import ctranslate2

import yaml
import numpy as np
import sounddevice as sd
import librosa
from faster_whisper import WhisperModel
import faiss
from lxml import etree
from sentence_transformers import SentenceTransformer
import requests

# -----------------------------
# Utils
# -----------------------------

def ensure_ollama_model_available(endpoint: str, model: str):
    """
    Tries to list models from Ollama. If the model isn't present or the server
    doesn't expose /api/tags, we still proceed (fallbacks may work), but we print hints.
    """
    url = endpoint.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code >= 400:
            print(f"[WARN] Ollama /api/tags returned {r.status_code}. The server may only expose /api/chat or /v1/*.")
            return
        data = r.json()
        names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        if names and model not in names:
            print(f"[WARN] Model '{model}' not listed by Ollama at {endpoint}.")
            print("      If this errors later, make sure it’s downloaded:  ollama pull " + model)
    except Exception as e:
        print(f"[WARN] Could not query Ollama /api/tags at {endpoint}: {e}")
        print("      If you see JSON 404s later, the server might be exposing only /api/chat or /v1/*.")
        
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def force_json(s: str):
    """
    Extract the first JSON object/array from a string (best effort).
    """
    m = re.search(r'([\{\[]).*$', s.strip(), flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON-looking content found")
    start_idx = m.start()
    candidate = s[start_idx:].strip()
    try:
        return json.loads(candidate)
    except Exception:
        last_brace = max(candidate.rfind("}"), candidate.rfind("]"))
        if last_brace != -1:
            try:
                return json.loads(candidate[:last_brace+1])
            except Exception:
                pass
        raise

def save_json(path, obj, pretty=False):
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            json.dump(obj, f, ensure_ascii=False)

def list_input_devices():
    devices = sd.query_devices()
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, d["name"], d["default_samplerate"], d["max_input_channels"]))
    return out

def resolve_input_device(dev_cfg):
    """
    dev_cfg can be:
      - None -> default input device
      - int  -> exact device index
      - str  -> case-insensitive substring to match device name
    Returns tuple (device_index_or_None, device_info_dict)
    """
    devices = sd.query_devices()
    # default input device index can be a pair (in,out) in sounddevice
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    if dev_cfg is None:
        idx = default_in if default_in is not None and default_in >= 0 else None
        info = sd.query_devices(idx, "input") if idx is not None else sd.query_devices(kind="input")
        return idx, info

    if isinstance(dev_cfg, int):
        info = sd.query_devices(dev_cfg, "input")
        return dev_cfg, info

    # string: match by substring
    cand = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and dev_cfg.lower() in d["name"].lower():
            cand.append((i, d))
    if cand:
        i, d = cand[0]
        return i, sd.query_devices(i, "input")

    # fallback to default input
    info = sd.query_devices(kind="input")
    return None, info

# -----------------------------
# Bible loading / indexing
# -----------------------------
def collect_xml_files(paths):
    """
    Given a list of paths (files/dirs/globs), return a sorted set of XMLs.
    Directories are searched recursively.
    """
    files = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for fp in pth.rglob("*"):
                name = fp.name.lower()
                if fp.is_file() and (name.endswith(".xml") or name.endswith(".osis.xml")):
                    files.append(str(fp))
        elif pth.is_file():
            name = pth.name.lower()
            if name.endswith(".xml") or name.endswith(".osis.xml"):
                files.append(str(pth))
        else:
            # allow simple glob patterns
            for fp in Path().glob(str(p)):
                if fp.is_file():
                    name = fp.name.lower()
                    if name.endswith(".xml") or name.endswith(".osis.xml"):
                        files.append(str(fp))
    return sorted(set(files))

# -- Entity fixes for non-XML HTML entities found in some Zefania files (e.g., NET)
_HTML_ENTITY_FIXES = {
    "&copy;": "©",
    "&nbsp;": " ",
    "&mdash;": "—",
    "&ndash;": "–",
    "&lsquo;": "‘",
    "&rsquo;": "’",
    "&ldquo;": "“",
    "&rdquo;": "”",
    "&hellip;": "…",
    # Add more if needed
}

def _preclean_entities(xml_text: str) -> str:
    # First, replace known bad HTML entities that aren't defined in XML
    for k, v in _HTML_ENTITY_FIXES.items():
        xml_text = xml_text.replace(k, v)
    # Leave standard XML entities (&lt; &gt; &amp; &apos; &quot;) untouched.
    return xml_text

def _parse_xml_string(xml_text: str):
    # Robust parser; recover=True lets lxml skip some well-formedness issues
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    return etree.fromstring(xml_text.encode("utf-8"), parser=parser)

def parse_zefania(xml_path, source_name=None):
    """
    Zefania XML schema:
      XMLBIBLE > BIBLEBOOK[bname,bnumber] > CHAPTER[cnumber] > VERS[vnumber]
    """
    if source_name is None:
        source_name = Path(xml_path).stem

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = _preclean_entities(raw)
    root = _parse_xml_string(raw)

    if root is None or (root.tag.lower() != "xmlbible" and not root.tag.lower().endswith("xmlbible")):
        return []  # Not Zefania

    verses = []
    # Zefania is usually no namespace, but handle possible namespaces
    # We'll access by local-name() to be safe.
    biblebooks = root.xpath(".//*[local-name()='BIBLEBOOK']")
    for bb in biblebooks:
        book = bb.get("bname") or bb.get("bsname") or bb.get("bnameR") or "UNKNOWN"
        chapters = bb.xpath(".//*[local-name()='CHAPTER']")
        for ch in chapters:
            chap = ch.get("cnumber") or ch.get("c", None)
            chap_int = int(chap) if chap and chap.isdigit() else None
            vers_nodes = ch.xpath(".//*[local-name()='VERS']")
            for v in vers_nodes:
                ver = v.get("vnumber") or v.get("v", None)
                ver_int = int(ver) if ver and ver.isdigit() else None
                text = " ".join(v.itertext()).strip()
                text = re.sub(r"\s+", " ", text)
                if not text:
                    continue
                ref = f"{book}.{chap}.{ver}" if (book and chap and ver) else f"{book}.{chap or 0}.{ver or 0}"
                verses.append({
                    "source": source_name,
                    "reference": ref,
                    "book": book,
                    "chapter": chap_int,
                    "verse": ver_int,
                    "text": text
                })
    return verses

def parse_osis(xml_path, source_name=None):
    """
    OSIS parser:
      <verse osisID="Book.Chapter.Verse">Text...</verse>
    """
    if source_name is None:
        source_name = Path(xml_path).stem

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = _preclean_entities(raw)
    root = _parse_xml_string(raw)
    if root is None:
        return []

    ns = root.nsmap.get(None) or root.nsmap.get("osis")
    nsmap = {"o": ns} if ns else {}
    verse_xpath = ".//o:verse" if ns else ".//verse"

    verses = []
    for v in root.xpath(verse_xpath, namespaces=nsmap):
        ref = v.get("osisID") or v.get("sID") or v.get("n") or "UNKNOWN"
        text = " ".join(v.itertext()).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        book, chap, ver = None, None, None
        if "." in ref:
            parts = ref.split(".")
            if len(parts) >= 3:
                book, chap, ver = parts[0], parts[1], parts[2]
        verses.append({
            "source": source_name,
            "reference": ref,
            "book": book,
            "chapter": int(chap) if chap and str(chap).isdigit() else None,
            "verse": int(ver) if ver and str(ver).isdigit() else None,
            "text": text
        })
    return verses

def parse_any_bible(xml_path):
    """
    Try Zefania first (your files), then OSIS.
    """
    # Quick sniff by looking for the string "<XMLBIBLE" to avoid double parsing.
    try:
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
    except Exception:
        head = ""
    if "<XMLBIBLE" in head or "<xmlbible" in head:
        verses = parse_zefania(xml_path, source_name=Path(xml_path).stem)
        if verses:
            return verses
    # Fallback to OSIS
    return parse_osis(xml_path, source_name=Path(xml_path).stem)

def load_bibles(xml_paths):
    """
    Accepts list of files/dirs/globs. Recursively collects XMLs.
    Parses Zefania or OSIS. De-duplicates by (source, reference).
    """
    xml_files = collect_xml_files(xml_paths)
    if not xml_files:
        print("[WARN] No XML files found from bible.xml_paths.")
        return []

    allv = []
    for p in xml_files:
        try:
            verses = parse_any_bible(p)
            allv.extend(verses)
            print(f"[INFO] Parsed {len(verses)} verses from {p}")
        except Exception as e:
            print(f"[ERROR] Failed to parse {p}: {e}")

    seen = set()
    dedup = []
    for v in allv:
        key = (v.get("source"), v["reference"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(v)

    print(f"[INFO] Total verses after dedup: {len(dedup)} from {len(xml_files)} file(s)")
    return dedup

def build_or_load_index(verses, meta_path, index_path, encoder_name):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        index = faiss.read_index(index_path)
        print("[INFO] Loaded FAISS index from disk.")
        return index, meta
    # Build
    print("[INFO] Building embeddings index (one-time). This can take a minute…")
    encoder = SentenceTransformer(encoder_name)
    texts = [v["text"] for v in verses]
    emb = encoder.encode(texts, normalize_embeddings=True, batch_size=512, convert_to_numpy=True)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, index_path)
    meta = {
        "embeddings_dim": d,
        "encoder": encoder_name,
        "verses_count": len(verses),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("[INFO] Index built and saved.")
    return index, meta

def retrieve_candidates(index, encoder, verses, query_text, k):
    qe = encoder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(qe.astype(np.float32), k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if int(idx) < 0:
            continue
        v = verses[int(idx)]
        out.append({
            "source": v.get("source"),
            "reference": v["reference"],
            "book": v["book"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "text": v["text"],
            "sim": float(score)
        })
    return out

# -----------------------------
# LLM Clients
# -----------------------------
class OllamaClient:
    def __init__(self, endpoint, model, temperature=0.2):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature

    def generate_json(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        r = requests.post(f"{self.endpoint}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        txt = r.json().get("response", "")
        return force_json(txt)

class OllamaClient:
    def __init__(self, endpoint, model, temperature=0.2, timeout=120):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.timeout = timeout

    # --- helpers ---
    def _post(self, path, payload):
        url = f"{self.endpoint}{path}"
        r = requests.post(url, json=payload, timeout=self.timeout)
        # For fallbacks, let caller see the exact status
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason} for url: {url}", response=r)
        return r.json()

    def _try_generate(self, prompt: str):
        # Classic Ollama /api/generate
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = self._post("/api/generate", payload)
        # Standard field is "response"
        txt = data.get("response", "")
        return txt

    def _try_chat(self, prompt: str):
        # Ollama /api/chat
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [{"role": "user", "content": prompt}],
        }
        data = self._post("/api/chat", payload)
        # Typical shape: {"message":{"role":"assistant","content":"..."}}
        msg = data.get("message", {}) if isinstance(data, dict) else {}
        txt = msg.get("content", "")
        return txt

    def _try_openai_compat(self, prompt: str):
        # OpenAI-compatible /v1/chat/completions (some launchers expose only this)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        data = self._post("/v1/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            return ""
        txt = (choices[0].get("message") or {}).get("content", "")
        return txt

    def generate_json(self, prompt: str):
        # Try /api/generate
        try:
            txt = self._try_generate(prompt)
            return force_json(txt)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code not in (404, 405):
                # Other HTTP errors: bail
                raise
            # else: fall through to next
        except Exception:
            # network or parse errors -> try next
            pass

        # Try /api/chat
        try:
            txt = self._try_chat(prompt)
            return force_json(txt)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code not in (404, 405):
                raise
        except Exception:
            pass

        # Try OpenAI-compatible endpoint
        txt = self._try_openai_compat(prompt)
        return force_json(txt)
    
def make_prompt(context_text, candidates):
    """
    Ask the LLM to rank and output top-5 JSON only.
    Include 'source' so model knows multiple translations may exist.
    """
    cand_lines = []
    for c in candidates:
        ref = c["reference"]
        t = c["text"]
        src = c.get("source", "unknown")
        cand_lines.append(
            f'{{"source":"{src}","reference":"{ref}","text":{json.dumps(t)}}}'
        )
    cands_blob = "[\n" + ",\n".join(cand_lines) + "\n]"

    prompt = f"""
Given the sermon transcript CONTEXT and a set of candidate Bible verses (possibly from multiple translations), pick the 5 most likely verses the speaker is either referencing or about to quote.

Rules:
- Consider both semantic meaning and common sermon flow.
- Prefer exact thematic matches and well-known follow-ups.
- Output strictly valid JSON with this shape:
{{
  "predictions": [
    {{"reference": "Book.Chapter.Verse", "score": 0.0_to_1.0, "reason": "short one-line why"}}
  ]
}}
- Exactly 5 items, sorted by descending score. Do not include verse text in the output.

CONTEXT:
\"\"\"{context_text[-12000:]}\"\"\"

CANDIDATES:
{cands_blob}
"""
    return prompt

def create_whisper_model_safe(model_size: str, compute_type: str):
    """
    Try to create a WhisperModel using the requested compute_type,
    then fall back through a safe list if unsupported.
    Good fallbacks for CPU/Metal are int8_float16 or int8.
    """
    # Try the requested type first, then fallbacks:
    fallbacks = [compute_type, "int8_float16", "int8", "float32"]

    last_err = None
    for ct in fallbacks:
        try:
            return WhisperModel(model_size, compute_type=ct)
        except ValueError as e:
            # Typical: "Requested float16 compute type, but the target device..."
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    # If everything failed, raise the last error
    if last_err:
        raise last_err
# -----------------------------
# Audio / Whisper
# -----------------------------
class MicTranscriber(threading.Thread):
    def __init__(self, cfg, out_queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.out_queue = out_queue
        self.running = True

        # ---- Whisper model (with your safe init) ----
        print("[INFO] Loading Whisper model…")
        requested_ct = self.cfg.get("compute_type", "float16")
        self.model = create_whisper_model_safe(self.cfg["model_size"], requested_ct)

        # ---- Audio config ----
        self.target_rate = int(self.cfg.get("sample_rate", 16000))
        self.chunk_seconds = float(self.cfg.get("chunk_seconds", 6))
        self.vad_filter = bool(self.cfg.get("vad_filter", True))
        self.dev_cfg = self.cfg.get("device", None)  # None | int | str
        self.min_words_per_push = int(self.cfg.get("min_words_per_push", 5))

        # Resolve device & native params
        self.device_index, self.device_info = resolve_input_device(self.dev_cfg)
        self.native_rate = int(self.device_info["default_samplerate"])
        max_in = int(self.device_info.get("max_input_channels", 1))
        self.input_channels = 1 if max_in >= 1 else max_in  # at least 1

        # Buffers
        self.buffer = np.zeros(0, dtype=np.float32)

        # Diagnostics
        dev_name = self.device_info["name"]
        dev_idx_str = "default" if self.device_index is None else str(self.device_index)
        print(f"[INFO] Using input device [{dev_idx_str}]: {dev_name} | "
              f"native_rate={self.native_rate} | channels={self.input_channels}")

    def _open_stream(self):
        """
        Open InputStream with native device rate (most reliable).
        We resample to target_rate inside the callback if needed.
        """
        return sd.InputStream(
            callback=self.callback,
            channels=self.input_channels,
            samplerate=self.native_rate,   # <-- open at native rate to avoid AUHAL -50
            device=self.device_index,
            dtype="float32",
            blocksize=0,                   # let CoreAudio choose
            latency="low"                  # can change to "high" if you still see xruns
        )

    def callback(self, indata, frames, time_info, status):
        if status:
            # Non-fatal (overruns/underruns); print once in a while if you want
            # print(status, file=sys.stderr)
            pass

        # Downmix to mono if needed
        if indata.ndim == 2 and indata.shape[1] > 1:
            audio = indata.mean(axis=1).astype(np.float32)
        else:
            audio = indata.squeeze().astype(np.float32)

        # Resample to the target rate if native != target
        if self.native_rate != self.target_rate:
            audio = librosa.resample(audio, orig_sr=self.native_rate, target_sr=self.target_rate)

        self.buffer = np.concatenate([self.buffer, audio])

        target_len = int(self.target_rate * self.chunk_seconds)
        if self.buffer.shape[0] >= target_len:
            chunk = self.buffer[:target_len]
            self.buffer = self.buffer[target_len:]
            self.process_chunk(chunk)

    def process_chunk(self, audio_chunk):
        segments, _ = self.model.transcribe(audio_chunk, vad_filter=self.vad_filter, language="en")
        text_parts = [seg.text.strip() for seg in segments if getattr(seg, "text", None)]
        if not text_parts:
            return
        text = " ".join(text_parts).strip()
        if text and len(text.split()) >= self.min_words_per_push:
            self.out_queue.put(text)

    def run(self):
        # Try opening the stream; if it fails, print devices and retry conservative
        try:
            with self._open_stream() as stream:
                print("[INFO] Mic stream opened. Speak when ready.")
                while self.running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[WARN] Failed to open input stream with [{self.device_info['name']}]: {e}")
            print("[INFO] Available input devices:")
            for i, name, rate, ch in list_input_devices():
                print(f"   [{i}] {name} | default_rate={rate} | max_in_ch={ch}")

            # Conservative retry: try default device, channels=1
            try:
                print("[INFO] Retrying with default input device & channels=1 …")
                default_info = sd.query_devices(kind="input")
                self.device_index = None
                self.native_rate = int(default_info["default_samplerate"])
                self.input_channels = 1
                with self._open_stream() as stream:
                    print("[INFO] Mic stream opened (fallback). Speak when ready.")
                    while self.running:
                        time.sleep(0.05)
            except Exception as ee:
                print(f"[FATAL] Could not open any input stream: {ee}")
                self.running = False

    def stop(self):
        self.running = False
# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    # Load Bible(s)
    verses = load_bibles(CFG["bible"]["xml_paths"])
    if not verses:
        print("[FATAL] No verses loaded. Check config.yaml bible.xml_paths and XML format.")
        sys.exit(1)

    # Build/Load embeddings index
    encoder = SentenceTransformer(CFG["bible"]["encoder"])
    index, _meta = build_or_load_index(
        verses, CFG["bible"]["meta_path"], CFG["bible"]["index_path"], CFG["bible"]["encoder"]
    )

    # LLM client
    provider = CFG["llm"]["provider"]
    if provider == "ollama":
        llm = OllamaClient(CFG["llm"]["endpoint"], CFG["llm"]["model"], CFG["llm"]["temperature"])
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("[FATAL] OPENAI_API_KEY not set.")
            sys.exit(1)
        llm = OpenAIClient(CFG["llm"]["openai_model"], CFG["llm"]["temperature"])
    else:
        print(f"[FATAL] Unknown LLM provider: {provider}")
        sys.exit(1)

    # Mic + Whisper
    q_text = queue.Queue()
    audio_cfg = {**CFG["audio"], **CFG["whisper"]}
    mt = MicTranscriber(audio_cfg, q_text)
    mt.start()

    context_text = ""
    max_chars = CFG["context"]["max_chars"]
    tail_chars = CFG["context"]["tail_chars"]
    k = CFG["bible"]["top_k_candidates"]

    out_path = CFG["output"]["write_file"]
    pretty = CFG["output"]["pretty_print"]
    print_console = CFG["output"]["print_to_console"]

    print("[INFO] Ready. Ctrl+C to stop.")
    try:
        while True:
            try:
                new_text = q_text.get(timeout=0.2)
            except queue.Empty:
                continue

            # Update rolling context
            context_text = (context_text + " " + new_text).strip() if context_text else new_text

            # Trim if needed
            if len(context_text) > max_chars:
                context_text = context_text[-tail_chars:]

            # Retrieve candidates by embeddings (bias to latest material)
            query_for_retrieval = context_text[-2000:]
            candidates = retrieve_candidates(index, encoder, verses, query_for_retrieval, k=k)

            # Ask LLM to pick top 5
            prompt = make_prompt(context_text, candidates)
            try:
                ranked = llm.generate_json(prompt)
            except Exception as e:
                print(f"[WARN] LLM JSON parse error: {e}")
                continue

            preds = ranked.get("predictions", [])[:5]
            payload = {
                "timestamp": now_iso(),
                "latest_transcript": new_text,
                "predictions": preds
            }

            if print_console:
                print(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None))

            if out_path:
                save_json(out_path, payload, pretty=pretty)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")
    finally:
        mt.stop()
        mt.join(timeout=1.0)

if __name__ == "__main__":
    main()