import json
import os
import subprocess

def download_videos():
    with open('how2qa_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_dir = 'How2QA_100_Videos'
    os.makedirs(output_dir, exist_ok=True)
    
    scripts_dir = r'C:\Miniconda3\envs\code\Scripts'
    yt_dlp_exe = os.path.join(scripts_dir, 'yt-dlp.exe')
    ffmpeg_path = r'C:\Miniconda3\envs\code\Lib\site-packages\static_ffmpeg\bin\win32\ffmpeg.exe'
    
    print(f"Starting download of 100 clips into {output_dir}...")
    
    for i, item in enumerate(data):
        vid = item['video_id']
        start = item['start_time']
        end = item['end_time']
        url = item['youtube_url']
        
        # Format filename
        filename = f"{vid}_{start}_{end}.mp4".replace(":", "_")
        output_file = os.path.join(output_dir, filename)
        
        if os.path.exists(output_file):
            print(f"[{i+1}/100] {filename} already exists.")
            continue
            
        print(f"[{i+1}/100] Downloading {vid} ({start}s - {end}s)...")
        
        # yt-dlp command to download specific section
        cmd = [
            yt_dlp_exe,
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "--ffmpeg-location", ffmpeg_path,
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "--quiet", "--no-warnings",
            "-o", output_file,
            url
        ]
        
        try:
            # We use check=False to continue even if one video fails
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Try fallback if best mp4 fails (e.g. only webm available)
                print(f"Best MP4 failed for {vid}, trying generic best...")
                cmd_fallback = [
                    yt_dlp_exe,
                    "--download-sections", f"*{start}-{end}",
                    "--force-keyframes-at-cuts",
                    "--ffmpeg-location", ffmpeg_path,
                    "--quiet", "--no-warnings",
                    "-o", output_file,
                    url
                ]
                subprocess.run(cmd_fallback)
        except Exception as e:
            print(f"Exception downloading {vid}: {e}")

    print("\nDownload process completed.")

if __name__ == "__main__":
    download_videos()
