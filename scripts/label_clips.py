"""在浏览器里给一段连续画面标动作。

帧数可选 16、32、48、64。标签为摔倒、步行、蹲下、跳跃、坐下、其他、躺下。
只把新标注写到新文件，不改原来的训练片段。
"""

from __future__ import annotations

import argparse
import json
import pickle
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from common import ROOT


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_ORDER = ("train", "val", "test")
WINDOW_CHOICES = (16, 32, 48, 64)
ACTIONS = (
    ("fall", "摔倒"),
    ("walk", "步行"),
    ("squat", "蹲下"),
    ("jump", "跳跃"),
    ("sit", "坐下"),
    ("other", "其他"),
    ("lie", "躺下"),
)
ACTION_IDS = {key for key, _name in ACTIONS}


def window_starts(n_frames: int, length: int) -> list[tuple[int, int, bool]]:
    """按所选帧数切段。相邻片段只重叠八分之一。"""
    if n_frames <= length:
        return [(0, n_frames, True)]
    stride = length - length // 8
    starts = range(0, n_frames - length + 1, stride)
    return [(start, start + length, False) for start in starts]


def list_frames(video_dir: Path) -> list[Path]:
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


class ClipLabeler:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.raw_root = (ROOT / "data" / "raw" / dataset).resolve()
        self.processed_dir = ROOT / "data" / "processed" / dataset
        self.ann_path = ROOT / "data" / "annotations" / dataset / "action_labels.json"
        self.old_ann_path = ROOT / "data" / "annotations" / dataset / "clip_labels.json"
        self.export_dir = ROOT / "data" / "processed" / f"{dataset}_action_labels"
        self.lock = threading.Lock()
        self.frame_cache: dict[str, list[Path]] = {}
        self.videos = self._load_videos()
        self.cache: dict[int, list[dict]] = {}
        self.annotations = self._load_annotations()
        self.old_binary = self._load_old_binary()

    def _video_dir(self, video_id: str) -> Path:
        path = (self.raw_root / video_id).resolve()
        if path != self.raw_root and self.raw_root not in path.parents:
            raise ValueError("video path escapes the dataset folder")
        return path

    def _frames(self, video_id: str) -> list[Path]:
        cached = self.frame_cache.get(video_id)
        if cached is None:
            cached = list_frames(self._video_dir(video_id))
            self.frame_cache[video_id] = cached
        return cached

    def _load_videos(self) -> list[dict]:
        splits_path = self.processed_dir / "video_splits.pkl"
        with splits_path.open("rb") as f:
            saved = pickle.load(f)
        videos = []
        for split in SPLIT_ORDER:
            for video_id in saved["splits"][split]:
                n_frames = len(self._frames(video_id))
                videos.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "n_frames": n_frames,
                        "old_label": 1 if "/fall/" in f"/{video_id}/" else 0,
                    }
                )
        return videos

    def segments(self, length: int) -> list[dict]:
        if length not in WINDOW_CHOICES:
            raise ValueError("帧数只能是 16、32、48、64")
        cached = self.cache.get(length)
        if cached is not None:
            return cached
        rows = []
        for video in self.videos:
            for start, end, short in window_starts(video["n_frames"], length):
                rows.append(
                    {
                        "split": video["split"],
                        "video_id": video["video_id"],
                        "old_label": video["old_label"],
                        "length": length,
                        "start": start,
                        "end": end,
                        "n_frames": video["n_frames"],
                        "short": short,
                    }
                )
        self.cache[length] = rows
        return rows

    def _load_annotations(self) -> dict[str, str]:
        if not self.ann_path.exists():
            return {}
        data = json.loads(self.ann_path.read_text(encoding="utf-8"))
        saved = data.get("labels", {})
        return {str(k): str(v) for k, v in saved.items() if str(v) in ACTION_IDS}

    def _load_old_binary(self) -> dict[str, int]:
        if not self.old_ann_path.exists():
            return {}
        data = json.loads(self.old_ann_path.read_text(encoding="utf-8"))
        saved = data.get("labels", {})
        return {str(k): int(v) for k, v in saved.items() if str(v) in {"0", "1"} or v in (0, 1)}

    def _key(self, length: int, video_id: str, start: int) -> str:
        return f"{length}\t{video_id}\t{start}"

    def _old_hint(self, split: str, video_id: str, length: int, start: int) -> int | None:
        if length != 32 or start % 16:
            return None
        key = f"{split}\t{video_id}\t{start // 16}"
        return self.old_binary.get(key)

    def _save_annotations(self) -> None:
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "windows": list(WINDOW_CHOICES),
            "actions": [{"id": key, "name": name} for key, name in ACTIONS],
            "rule": "一段画面选一个动作：摔倒、步行、蹲下、跳跃、坐下、其他、躺下。",
            "labels": self.annotations,
        }
        temp = self.ann_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.ann_path)

    def summary(self, length: int) -> dict:
        counts = {key: 0 for key, _name in ACTIONS}
        labeled = 0
        with self.lock:
            for row in self.segments(length):
                value = self.annotations.get(self._key(length, row["video_id"], row["start"]))
                if value is None:
                    continue
                labeled += 1
                counts[value] += 1
        return {"dataset": self.dataset, "length": length, "total": len(self.segments(length)), "labeled": labeled, "counts": counts}

    def clip_list(self, length: int) -> list[dict]:
        with self.lock:
            rows = []
            for index, row in enumerate(self.segments(length)):
                item = dict(row)
                item["index"] = index
                item["action"] = self.annotations.get(self._key(length, row["video_id"], row["start"]))
                item["old_binary"] = self._old_hint(row["split"], row["video_id"], length, row["start"])
                rows.append(item)
            return rows

    def frame_urls(self, length: int, index: int) -> dict:
        row = self.segments(length)[index]
        frames = self._frames(row["video_id"])
        urls = []
        for frame_index, path in enumerate(frames[row["start"] : row["end"]], start=row["start"]):
            rel = path.resolve().relative_to(self.raw_root).as_posix()
            urls.append({"frame": frame_index, "url": f"/media/{self.dataset}/{rel}"})
        return {"index": index, "length": length, "frames": urls}

    def media_file(self, dataset: str, rel: str) -> Path:
        if dataset != self.dataset:
            raise ValueError("dataset mismatch")
        path = (self.raw_root / rel).resolve()
        if path != self.raw_root and self.raw_root not in path.parents:
            raise ValueError("media path escapes the dataset folder")
        if not path.is_file():
            raise FileNotFoundError(rel)
        return path

    def set_label(self, length: int, index: int, action: str | None) -> dict:
        if action is not None and action not in ACTION_IDS:
            raise ValueError("未知标签")
        with self.lock:
            row = self.segments(length)[index]
            key = self._key(length, row["video_id"], row["start"])
            if action is None:
                self.annotations.pop(key, None)
            else:
                self.annotations[key] = action
            self._save_annotations()
        return self.summary(length)

    def export(self) -> dict:
        with self.lock:
            labels = dict(self.annotations)
        items = []
        for length in WINDOW_CHOICES:
            for row in self.segments(length):
                action = labels.get(self._key(length, row["video_id"], row["start"]))
                if action is None:
                    continue
                items.append(
                    {
                        "split": row["split"],
                        "video_id": row["video_id"],
                        "length": length,
                        "start": row["start"],
                        "end": row["end"],
                        "action": action,
                    }
                )
        self.export_dir.mkdir(parents=True, exist_ok=True)
        report = {"export_dir": str(self.export_dir.relative_to(ROOT)), "labeled": len(items)}
        (self.export_dir / "labeled.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.export_dir / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>标注动作</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif; background: #121417; color: #f2f4f8; }
  header { display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 14px 18px; border-bottom: 1px solid #2a2f38; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  p { margin: 0; color: #b7c0cc; line-height: 1.45; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; padding: 16px; }
  .stage { background: #000; border-radius: 12px; min-height: 420px; display: flex; align-items: center; justify-content: center; position: relative; }
  img { max-width: 100%; max-height: 68vh; display: block; }
  .hud { position: absolute; left: 12px; bottom: 12px; background: rgba(0,0,0,.62); padding: 6px 10px; border-radius: 8px; }
  .side, .bar { background: #1b2028; border-radius: 12px; padding: 14px; }
  .bar { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  button, select { border: 0; border-radius: 8px; padding: 10px 14px; font-size: 15px; background: #2d3644; color: inherit; cursor: pointer; }
  button.on { outline: 2px solid #84adff; }
  button.primary { background: #175cd3; }
  .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .meta { display: grid; gap: 8px; margin: 12px 0; }
  .meta div { background: #12161d; border-radius: 8px; padding: 8px 10px; }
  .k { color: #93a0b3; font-size: 12px; }
  .strip { display: flex; gap: 6px; overflow-x: auto; margin-top: 10px; }
  .strip button { padding: 0; background: #000; }
  .strip img { height: 52px; width: 72px; object-fit: cover; opacity: .55; }
  .strip button.on img { opacity: 1; outline: 2px solid #84adff; }
  .toast { position: fixed; right: 16px; bottom: 16px; background: #053321; padding: 10px 14px; border-radius: 8px; display: none; }
</style>
</head>
<body>
<header>
  <div>
    <h1>给一段连续画面标动作</h1>
    <p>帧数可选 16、32、48、64。64 帧更容易看完一个人从摔倒到躺在地上。一段只选一个最主要的动作。标注写到新文件，原来的训练片段不会被改掉。</p>
  </div>
  <div id="progress">正在读取片段…</div>
</header>
<main>
  <section>
    <div class="stage">
      <img id="picture" alt="当前画面" />
      <div class="hud" id="hud">-</div>
    </div>
    <div class="strip" id="strip"></div>
  </section>
  <aside class="side">
    <div class="meta" id="meta"></div>
    <div class="actions" id="actions"></div>
    <button id="clear-label" style="margin-top:8px;width:100%">清除这条标注（按退格）</button>
  </aside>
  <section class="bar">
    <button id="prev">上一段</button>
    <button id="play">暂停</button>
    <button id="next">下一段</button>
    <label>帧数 <select id="length"><option>16</option><option>32</option><option>48</option><option selected>64</option></select></label>
    <label>速度 <select id="speed"><option>6</option><option selected>10</option><option>15</option><option>24</option></select> 帧/秒</label>
    <label>看哪一份 <select id="split"><option value="all">全部</option><option value="train">训练集</option><option value="val">验证集</option><option value="test">测试集</option></select></label>
    <label><input id="todo" type="checkbox" checked /> 只看还没标的</label>
    <button class="primary" id="export">导出新标注</button>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const ACTIONS = [
  ["fall", "摔倒", "1"],
  ["walk", "步行", "2"],
  ["squat", "蹲下", "3"],
  ["jump", "跳跃", "4"],
  ["sit", "坐下", "5"],
  ["other", "其他", "6"],
  ["lie", "躺下", "7"],
];
const ACTION_NAME = Object.fromEntries(ACTIONS.map(([id, name]) => [id, name]));
const SPLIT_NAME = {train: "训练集", val: "验证集", test: "测试集"};
const OLD_NAME = {1: "摔倒视频", 0: "日常活动"};
const OLD_BINARY = {1: "上次标过：有摔倒", 0: "上次标过：没有摔倒"};
let clips = [];
let view = [];
let pos = 0;
let frames = [];
let framePos = 0;
let playing = true;
let timer = null;
let length = 64;

const picture = document.getElementById("picture");
const hud = document.getElementById("hud");
const meta = document.getElementById("meta");
const strip = document.getElementById("strip");
const progress = document.getElementById("progress");
const toast = document.getElementById("toast");
const actionBox = document.getElementById("actions");
ACTIONS.forEach(([id, name, key]) => {
  const btn = document.createElement("button");
  btn.textContent = `${name}（按 ${key}）`;
  btn.dataset.action = id;
  btn.onclick = () => mark(id);
  actionBox.appendChild(btn);
});

function showToast(text) {
  toast.textContent = text;
  toast.style.display = "block";
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { toast.style.display = "none"; }, 1600);
}
function current() { return view[pos]; }
function clipKey(c) { return `${c.length}|${c.video_id}|${c.start}`; }
function refreshProgress() {
  const labeled = clips.filter(c => c.action).length;
  const parts = ACTIONS.map(([id, name]) => `${name} ${clips.filter(c => c.action === id).length}`).join("，");
  progress.textContent = `当前 ${length} 帧：已标 ${labeled} / ${clips.length}。${parts}`;
}
function applyFilter() {
  const split = document.getElementById("split").value;
  const todo = document.getElementById("todo").checked;
  const currentKey = current() ? clipKey(current()) : "";
  view = clips.filter(c => (split === "all" || c.split === split) && (!todo || !c.action));
  pos = Math.max(0, view.findIndex(c => clipKey(c) === currentKey));
  if (!view.length) {
    picture.removeAttribute("src");
    hud.textContent = "这一筛选下没有片段了";
    meta.innerHTML = "<div>可以取消「只看还没标的」，或换帧数、换一份数据。</div>";
    strip.innerHTML = "";
    return;
  }
  loadClip();
}
async function loadClips() {
  length = Number(document.getElementById("length").value);
  const data = await (await fetch(`/api/clips?length=${length}`)).json();
  clips = data.clips;
  refreshProgress();
  applyFilter();
}
async function loadClip() {
  const clip = current();
  if (!clip) return;
  const data = await (await fetch(`/api/frames?length=${clip.length}&index=${clip.index}`)).json();
  frames = data.frames;
  framePos = 0;
  renderMeta();
  renderStrip();
  showFrame();
  play();
}
function renderMeta() {
  const c = current();
  const mine = c.action ? ACTION_NAME[c.action] : "还没标";
  const span = c.short ? `整段只有 ${c.end - c.start} 帧，短于所选的 ${c.length} 帧` : `原视频第 ${c.start + 1} 到 ${c.end} 帧，共 ${c.end - c.start} 帧`;
  const rows = [
    ["位置", `第 ${pos + 1} / ${view.length} 段`],
    ["所属", SPLIT_NAME[c.split]],
    ["视频", c.video_id],
    ["这段画面", span],
    ["原来的整段视频", OLD_NAME[c.old_label]],
    ["你的标注", mine],
  ];
  if (c.old_binary === 0 || c.old_binary === 1) rows.push(["以前的两类标注", OLD_BINARY[c.old_binary]]);
  meta.innerHTML = rows.map(([k, v]) => `<div><div class="k">${k}</div>${v}</div>`).join("");
  [...actionBox.children].forEach(btn => btn.classList.toggle("on", btn.dataset.action === c.action));
}
function renderStrip() {
  strip.innerHTML = "";
  const step = Math.max(1, Math.floor(frames.length / 12));
  frames.forEach((frame, i) => {
    if (i % step !== 0 && i !== frames.length - 1) return;
    const btn = document.createElement("button");
    btn.innerHTML = `<img src="${frame.url}" alt="" />`;
    btn.onclick = () => { framePos = i; showFrame(); };
    btn.dataset.i = String(i);
    strip.appendChild(btn);
  });
}
function showFrame() {
  if (!frames.length) return;
  const frame = frames[framePos];
  picture.src = frame.url;
  hud.textContent = `这段第 ${framePos + 1} / ${frames.length} 帧，原视频第 ${frame.frame + 1} 帧`;
  [...strip.children].forEach(btn => btn.classList.toggle("on", Number(btn.dataset.i) === framePos));
}
function play() {
  clearInterval(timer);
  playing = true;
  document.getElementById("play").textContent = "暂停";
  const fps = Number(document.getElementById("speed").value);
  timer = setInterval(() => {
    framePos = (framePos + 1) % frames.length;
    showFrame();
  }, 1000 / fps);
}
function pause() {
  clearInterval(timer);
  playing = false;
  document.getElementById("play").textContent = "播放";
}
async function mark(action) {
  const clip = current();
  if (!clip) return;
  const res = await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({length: clip.length, index: clip.index, action})});
  if (!res.ok) { showToast("保存失败"); return; }
  clip.action = action;
  clips[clip.index].action = action;
  refreshProgress();
  showToast(action ? "已保存" : "已清除");
  if (document.getElementById("todo").checked) applyFilter();
  else if (pos < view.length - 1) { pos += 1; loadClip(); }
  else renderMeta();
}
document.getElementById("clear-label").onclick = () => mark(null);
document.getElementById("prev").onclick = () => { if (pos > 0) { pos -= 1; loadClip(); } };
document.getElementById("next").onclick = () => { if (pos < view.length - 1) { pos += 1; loadClip(); } };
document.getElementById("play").onclick = () => playing ? pause() : play();
document.getElementById("speed").onchange = () => { if (playing) play(); };
document.getElementById("length").onchange = () => loadClips().catch(() => { progress.textContent = "片段读取失败"; });
document.getElementById("split").onchange = applyFilter;
document.getElementById("todo").onchange = applyFilter;
document.getElementById("export").onclick = async () => {
  const res = await fetch("/api/export", {method: "POST"});
  const data = await res.json();
  if (!res.ok) { showToast("导出失败"); return; }
  showToast("已导出");
  alert(`已导出 ${data.labeled} 条标注到 ${data.export_dir}。原来的训练片段没有改动。`);
};
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, textarea")) return;
  const found = ACTIONS.find(([, , key]) => key === event.key);
  if (found) mark(found[0]);
  else if (event.key === "ArrowLeft") document.getElementById("prev").click();
  else if (event.key === "ArrowRight") document.getElementById("next").click();
  else if (event.key === " ") { event.preventDefault(); document.getElementById("play").click(); }
  else if (event.key === "Backspace") mark(null);
});
loadClips().catch(() => { progress.textContent = "片段读取失败"; });
</script>
</body>
</html>
"""


def _length_from(query: dict, default: int = 64) -> int:
    raw = query.get("length", [str(default)])[0]
    length = int(raw)
    if length not in WINDOW_CHOICES:
        raise ValueError("帧数只能是 16、32、48、64")
    return length


def make_handler(labeler: ClipLabeler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == "/":
                    self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/clips":
                    length = _length_from(query)
                    self._json(200, {"summary": labeler.summary(length), "clips": labeler.clip_list(length)})
                    return
                if parsed.path == "/api/frames":
                    length = _length_from(query)
                    index = int(query.get("index", ["-1"])[0])
                    self._json(200, labeler.frame_urls(length, index))
                    return
                if parsed.path.startswith("/media/"):
                    parts = [unquote(part) for part in parsed.path.split("/")[2:]]
                    dataset, rel = parts[0], "/".join(parts[1:])
                    path = labeler.media_file(dataset, rel)
                    kind = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                    self._send(200, path.read_bytes(), kind)
                    return
                self._json(404, {"error": "not found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                size = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(size) if size else b"{}"
                if parsed.path == "/api/label":
                    data = json.loads(raw.decode("utf-8"))
                    action = data.get("action")
                    summary = labeler.set_label(int(data["length"]), int(data["index"]), None if action is None else str(action))
                    self._json(200, summary)
                    return
                if parsed.path == "/api/export":
                    self._json(200, labeler.export())
                    return
                self._json(404, {"error": "not found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="在浏览器里标注一段画面的动作")
    parser.add_argument("--dataset", default="URFD")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    labeler = ClipLabeler(args.dataset)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(labeler))
    url = f"http://{args.host}:{args.port}"
    print(f"已读取 {len(labeler.videos)} 段视频。在浏览器打开 {url}")
    print("帧数可选 16、32、48、64。新标注写在 data/annotations 里的动作文件，不改原来的训练片段。")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭标注页面")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
