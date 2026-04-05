import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp


def parse_args():
    parser = argparse.ArgumentParser(description="Download YouTube video as 480p mp4 into videos directory.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--videos-dir", type=Path, default=None, help="Target videos directory (default: VIDEOS_DIR env or ./videos)")
    parser.add_argument("--output-name", default=None, help="Optional output base name (without extension)")
    return parser.parse_args()


def build_output_template(videos_dir, output_name):
    if output_name:
        safe = output_name.strip().replace("/", "_").replace("\\", "_")
        return str(videos_dir / f"{safe}.%(ext)s")
    return str(videos_dir / "%(title).180B [%(id)s].%(ext)s")


def main():
    load_dotenv()
    args = parse_args()
    videos_dir = args.videos_dir or Path(os.getenv("VIDEOS_DIR", "videos"))
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    with yt_dlp.YoutubeDL({
        "format": "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best[height<=480]/best",
        "outtmpl": build_output_template(videos_dir, args.output_name),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
    }) as ydl:
        ydl.download([args.url])
    
    print(f"[done] saved into: {videos_dir.as_posix()}")


if __name__ == "__main__":
    main()
