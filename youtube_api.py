from googleapiclient.discovery import build

# 발급받은 본인 YouTube API 키 넣기
API_KEY = "AIzaSyB4ml-OdaqLb3J3sgpMScKmMgOYWPxdVf8"

youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_youtube(query, max_results=3):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response['items']:
        videos.append({
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })
    return videos
