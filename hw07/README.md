# ITCS 8101 NLP — HW07: Restaurant Reservation Chatbot (MCP + Gemini)

Homework 7 for ITCS 8101 (Natural Language Processing), UNC Charlotte, Spring 2026.

A task-oriented dialog system that helps a user find a restaurant and book a table.
It has two parts that communicate over the
[Model Context Protocol](https://modelcontextprotocol.io/):

- **MCP server** (`chatbot_server/mcp_server.py`): a [FastMCP](https://gofastmcp.com)
  server that owns the restaurant database and exposes these tools:
  - `create_vdb_for_location(City, State)`: narrows the search to one city
  - `vdb_search_restaurants(dialog_state, query, k)`: filters restaurants by the
    current dialog state (cuisine, name, extra constraints)
  - `get_restaurant_strs(restaurant_indexes)`: returns restaurant descriptions
  - `make_reservation(restaurant_id, date, time)`: appends the booking to `reservations.csv`
- **Host / dialog manager** (`chatbot_host/mcp_host.py`): uses Gemini
  (`gemini-2.5-flash-lite` by default) with function calling. It tracks a dialog
  state (`City`, `State`, `Cuisine`, `Name`, `Misc`), asks clarifying questions
  (for example, a city name that exists in several states), and calls the MCP tools.

`chatbot_host/ChatbotTest.ipynb` runs four test dialogs: cuisine only, extra
constraints only, an ambiguous city name, and a user who changes their mind.

## Repository layout

```
chatbot_server/
  mcp_server.py
  reservations.csv           bookings written by make_reservation
  data/
    restaurants.md / .pdf    description of how the restaurant database was built
    reservations.csv
chatbot_host/
  mcp_host.py
  ChatbotTest.ipynb
.env.example
```

### Data not included

`chatbot_server/data/restaurants.json` (140 MB) is not in this repository. The
server needs it to run. It is a sample of the
[Yelp Academic Dataset](https://www.yelp.com/dataset), which is too large for GitHub
and whose license does not allow redistribution; `data/restaurants.md` describes
how it was sampled. The precomputed embedding files from development
(`data/embeddings/`, about 790 MB) are not included either.

## Setup and run

```bash
pip install -r requirements.txt
cp .env.example .env        # then set GEMINI_API_KEY
```

1. Start the MCP server (HTTP, port 8000):

   ```bash
   cd chatbot_server
   fastmcp run mcp_server.py --transport http --port 8000
   ```

2. In a second terminal, start the chatbot CLI:

   ```bash
   cd chatbot_host
   python mcp_host.py
   ```

Environment variables: `GEMINI_API_KEY` (required), `GEMINI_MODEL`
(default `gemini-2.5-flash-lite`), `MCP_URL` (default `http://localhost:8000/mcp`).
