# Slack Agent Example

Agentic RAG agent that interacts with Slack channel messages and search capabilities.

## 1. Create a Slack App

1. Go to [Slack API: Your Apps](https://api.slack.com/apps) and click **Create New App**.
2. Choose **From scratch** and give your app a name and workspace.

## 2. Set Required OAuth Scopes

Under **OAuth & Permissions**, add the following **Bot Token Scopes**:
- `app_mentions:read`
- `chat:write`
- `channels:history`
- `channels:read`
- `groups:history`
- `im:history`
- `im:read`
- `im:write`
- `mpim:history`
- `users:read`

(You may need fewer/more depending on your use case.)

## 2a. (Optional) User Token Scopes

If your agent needs to search workspace content or act on behalf of users, add the following **User Token Scope**:

- `search:read` — Search a workspace’s content

User Token Scopes allow your app to access user data and perform actions as the user who authorizes them. You will need to implement the OAuth flow to obtain a **User OAuth Token** if your agent requires these capabilities. See Slack's [OAuth documentation](https://api.slack.com/authentication/oauth-v2) for details.

## 3. Install the App and Get Tokens

1. Go to **OAuth & Permissions** and click **Install App to Workspace**.
2. Copy the **Bot User OAuth Token** (starts with `xoxb-`).

## 4. Set Environment Variables

Create a `.env` file in this directory with the following:

```
SLACK_BOT_TOKEN=xoxb-...      # Required for all Slack API operations (bot token)
SLACK_USER_TOKEN=xoxp-...     # Required only for search functionality (user token with search:read scope)
```

- `SLACK_BOT_TOKEN` is required for basic channel and message operations. You get this from the "Bot User OAuth Token" after installing your app.
- `SLACK_USER_TOKEN` is only required if you want to use the search functionality (e.g., searching messages in channels). This must be a User OAuth Token (starts with `xoxp-`) with the `search:read` user scope. You must implement the OAuth flow to obtain this token on behalf of a user. See Slack's [OAuth documentation](https://api.slack.com/authentication/oauth-v2) for details.

Or export these variables in your shell before running.

## 5. Run the Agent

Install dependencies (if needed):
```
pip install -r requirements.txt
```

Then run:
```
python agent.py
```

## Troubleshooting
- Make sure your app is invited to the channels you want it to listen to.
- Reinstall the app if you change scopes.
---
