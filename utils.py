"""
Multiple utilities
"""
import yt_dlp

def youtube_ids(file_path="data/Youtube_ID.txt"):
    with open(file_path) as f:
        for line in f.readlines():
            youtube_id = line.strip().split("=")[-1]
            yield youtube_id


def youtube_urls(file_path="data/Youtube_ID.txt"):
    with open(file_path) as f:
        for line in f.readlines():
            youtube_url = line.strip()
            yield youtube_url



# For more details check: https://github.com/yt-dlp/yt-dlp#embedding-examples

ydl_opts = {
    # 'format': 'm4a/bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    # 'postprocessors': [{  # Extract audio using ffmpeg
    #     'key': 'FFmpegExtractAudio',
    #     'preferredcodec': 'm4a',
    # }]
    #  'postprocessors': [{  # Extract audio using ffmpeg
    #     'key': 'FFmpegVideoConvertor',
    #     'preferedformat': 'mp4',
    # }],

    # TODO: Add the following post processors
    # MoveFilesAfterDownloadPP
    # FFmpegVideoRemuxerPP

    "concurrent_fragments": 100,

    # 'merge-output-format': "mp4"


}

def format_selector(ctx):
    """ Select the best video and the best audio that won't result in an mkv.
    NOTE: This is just an example and does not handle all cases """

    # formats are already sorted worst to best
    formats = ctx.get('formats')[::-1]

    # acodec='none' means there is no audio
    best_video = next(f for f in formats
                      if f['vcodec'] != 'none' and f['acodec'] == 'none')

    # find compatible audio extension
    audio_ext = {'mp4': 'm4a', 'webm': 'webm'}[best_video['ext']]
    # vcodec='none' means there is no video
    best_audio = next(f for f in formats if (
        f['acodec'] != 'none' and f['vcodec'] == 'none' and f['ext'] == audio_ext))

    # These are the minimum required fields for a merged format
    yield {
        'format_id': f'{best_video["format_id"]}+{best_audio["format_id"]}',
        'ext': best_video['ext'],
        'requested_formats': [best_video, best_audio],
        # Must be + separated list of protocols
        'protocol': f'{best_video["protocol"]}+{best_audio["protocol"]}'
    }


# ydl_opts = {
#     'format': format_selector,
# }

def download_video(youtube_url):
    # ydl_opts = {}
    print(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def download_videos(urls):
    print(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
