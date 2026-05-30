from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
def get_video_id(url: str)-> str:

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format")

def fetch_comments(video_id: str, max_comments: int = 100) -> list:
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            textFormat="plainText",
            order="relevance"
        )

        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": top_comment["textDisplay"],
                    "likes": top_comment["likeCount"],
                    "date": top_comment["publishedAt"]
                })
            request = youtube.commentThreads().list_next(request, response)
        return comments[:max_comments]
    except Exception as e:
        raise ValueError(f"Error fetching comments: {str(e)}")

def fetch_comments_from_url(url: str, max_comments: int = 100) -> list:
    video_id = get_video_id(url)
    comments =fetch_comments(video_id, max_comments)
    return comments

             