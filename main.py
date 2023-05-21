import utils
import concurrent.futures

# for youtube_id, _ in zip(utils.youtube_ids(), range(10)):
#     print(youtube_id)

with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
    futures = []
    for youtube_url, _ in zip(utils.youtube_ids(), range(20)):
        print(youtube_url)
        futures.append(executor.submit(utils.download_video, youtube_url=youtube_url))
        # break

    for future in concurrent.futures.as_completed(futures):
        print(future.result())

# utils.download_videos([url for url, _ in zip(utils.youtube_urls(), range(10))])

