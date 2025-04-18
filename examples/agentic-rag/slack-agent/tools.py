import os
from slack_bolt import App
import pixeltable as pxt

from dotenv import load_dotenv

load_dotenv()

# Initialize Slack app with bot token
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

def get_recent_messages(channel_id, limit=5):
    """
    Fetch the most recent messages from a specified Slack channel.
    
    Args:
        channel_id (str): The ID of the channel to fetch messages from
        limit (int): Number of messages to retrieve (default: 5)
    
    Returns:
        list: List of message dictionaries, most recent first
    """
    try:
        # Attempt to join the channel first
        app.client.conversations_join(channel=channel_id)
        print(f"Joined channel with ID: {channel_id}")
    except Exception as e:
        print(f"Error joining channel: {e}")
        
    try:
        response = app.client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        print(response)
        return [{"text": msg["text"], "user": msg["user"], "ts": msg["ts"]} for msg in response["messages"]]
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

def list_channels():
    """
    Fetch a list of all public channels in the Slack workspace.

    Returns:
        dict: Dictionary with channel names as keys and IDs as values
    """
    try:
        response = app.client.conversations_list(
            types="public_channel"
        )
        return {channel["name"]: channel["id"] for channel in response["channels"]}
    except Exception as e:
        print(f"Error fetching channels: {e}")
        return {}

def get_username(user_id):
    """
    Fetch the username for a given user ID.
    
    Args:
        user_id (str): The ID of the user to look up
    
    Returns:
        str: The username if found, otherwise 'Unknown User'
    """
    try:
        response = app.client.users_info(user=user_id)
        if response['ok']:
            return response['user']['real_name']
        return 'Unknown User'
    except Exception as e:
        print(f"Error fetching username: {e}")
        return 'Unknown User'

def format_conversation(messages):
    """
    Format a list of messages into a conversation string.
    
    Args:
        messages (list): List of message dictionaries
    
    Returns:
        str: Formatted conversation string with timestamp, real name, and text
    """
    conversation = ""
    for msg in messages:
        username = get_username(msg['user'])
        conversation += f"{msg['ts']}: {username}: {msg['text']}\n"
    return conversation

def get_channel_messages(channel_name):
    channels = list_channels()
    channel_id = channels.get(channel_name, None)
    if channel_id:
        messages = get_recent_messages(channel_id)
        conversation = format_conversation(messages)
        return conversation
    else:
        return f"Channel '{channel_name}' not found"

@pxt.udf
def search_channel_messages(channel_name: str, keyword: str) -> str:
    """
    Search for messages containing a specific keyword in a Slack channel.

    Args:
        channel_name (str): Name of the channel to search in
        keyword (str): Keyword or phrase to search for
        limit (int): Maximum number of matching messages to return (default: 5)
    
    Returns:
        str: Formatted string of matching messages, or error message if channel not found
    """
    channels = list_channels()
    channel_id = channels.get(channel_name, None)
    if channel_id:
        try:
            # Join the channel if not already joined
            app.client.conversations_join(channel=channel_id)
            print(f"Joined channel with ID: {channel_id}")
        except Exception as e:
            print(f"Error joining channel: {e}")
            
        try:
            # Use Slack's search.messages API to find messages with the keyword
            query = f"{keyword} in:#{channel_name}"
            response = app.client.search_messages(token=os.environ.get("SLACK_USER_TOKEN"), query=query, count=10)
            print(response)
            if response['ok'] and 'messages' in response and 'matches' in response['messages']:
                matches = response['messages']['matches'][:10]
                return format_conversation([{
                    'text': match['text'],
                    'user': match['user'],
                    'ts': match['ts']
                } for match in matches])
            else:
                return f"No messages found containing '{keyword}' in channel '{channel_name}'"
        except Exception as e:
            print(f"Error searching messages: {e}")
            if str(e).find('not_allowed_token_type') != -1:
                return f"Error searching for '{keyword}' in channel '{channel_name}'"
            return f"Error searching for '{keyword}' in channel '{channel_name}': {str(e)}"
    else:
        return f"Channel '{channel_name}' not found"
