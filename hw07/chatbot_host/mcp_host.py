from google import genai
import json
import re
from typing import List, Dict, Any
import asyncio
import os

import fastmcp

MCP_URL = os.environ.get("MCP_URL", "http://localhost:8000/mcp")
mcp_client = fastmcp.Client(MCP_URL)

from pydantic import BaseModel, Field

class DialogState(BaseModel):
    """Pydantic model for tracking dialog state."""
    
    City: str = Field(default='-', description="City name")
    State: str = Field(default='-', description="State abbreviation (2 letters)")
    Cuisine: str = Field(default='-', description="Cuisine type(s)")
    Name: str = Field(default='-', description="Restaurant name(s)")
    Misc: str = Field(default='-', description="Additional constraints")
    
    model_config = {'extra': 'forbid'}
    
    def update(self, **kwargs):
        """Update dialog state fields with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value:
                setattr(self, key, value)
    
    def get_state(self):
        """Return a dictionary copy of the current state."""
        return {
            'City': self.City,
            'State': self.State,
            'Cuisine': self.Cuisine,
            'Name': self.Name,
            'Misc': self.Misc
        }
    
    def to_json(self):
        """Convert dialog state to formatted JSON string."""
        return json.dumps(self.get_state(), indent=2)
        
    
def update_dialog_state(dialog_state: DialogState,
                        City = None, State = None, Cuisine = None, Name = None, Misc = None):
    """
    Update the dialog state with new preference values.
    
    This is a wrapper around DialogState.update() that allows
    updating any combination of dialog state fields.
    
    Parameters
    ----------
    dialog_state : DialogState
        Dialog state object to update
    City : str, optional
        City name (default: None, no update)
    State : str, optional
        State abbreviation, must be 2 letters (default: None, no update)
    Cuisine : str, optional
        Cuisine type(s), can be multiple (default: None, no update)
    Name : str, optional
        Restaurant name(s), can be multiple (default: None, no update)
    Misc : str, optional
        Any additional constraints (default: None, no update)
    
    Returns
    -------
    None
        Function has no explicit return value
    
    Behavior
    --------
    - Calls dialog_state.update() with provided parameters
    - Only updates fields where value is not None
    - Empty strings are passed through (update() will ignore them)
    - Non-existent keys are ignored by update()
    
    Update Rules
    ------------
    - Use "-" to explicitly clear a field
    
    Examples
    --------
    >>> ds = DialogState()
    >>> update_dialog_state(ds, City="Chicago", State="IL")
    >>> print(ds.state)
    {'City': 'Chicago', 'State': 'IL', 'Cuisine': '', 'Name': '', 'Misc': ''}
    
    >>> update_dialog_state(ds, Cuisine="Mexican", Misc="outdoor seating")
    >>> print(ds.state)
    {'City': 'Chicago', 'State': 'IL', 'Cuisine': 'Mexican', 'Name': '', 'Misc': 'outdoor seating'}
    
    >>> # Reset a field
    >>> update_dialog_state(ds, Cuisine="-")
    >>> print(ds.state['Cuisine'])
    '-'
    
    >>> # None doesn't update
    >>> update_dialog_state(ds, City=None, State="NY")
    >>> print(ds.state['City'])
    'Chicago'  # Still Chicago
    """
    
    dialog_state.update(City = City, State = State, Cuisine = Cuisine, Name = Name, Misc = Misc)
    return {"status": "success"}



from dotenv import load_dotenv, find_dotenv
from google.genai import types
from google.genai.errors import ClientError

_ = load_dotenv(find_dotenv())
_GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key = _GEMINI_API_KEY) if _GEMINI_API_KEY else None
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

STATE_ALIASES = {
    "alabama": "AL", "al": "AL",
    "florida": "FL", "fl": "FL",
    "indiana": "IN", "in": "IN",
    "new jersey": "NJ", "nj": "NJ",
    "pennsylvania": "PA", "pa": "PA",
    "tennessee": "TN", "tn": "TN",
}

KNOWN_CUISINES = [
    "thai", "italian", "mexican", "chinese", "japanese", "sushi",
    "pizza", "bbq", "barbecue", "indian", "korean", "vietnamese",
    "american", "seafood", "breakfast", "sandwich", "burger",
]

initial_prompt = """\
You are a helpful restaurant reservation assistant. Your job is to help users find the perfect restaurant.

DIALOG STATE MANAGEMENT:
- Maintain a dialog state with: Location, Cuisine, Name, Misc
- Extract and update these fields from user messages

PRIORITY : Updating the Dialog State: Prioritize obtaining city and state 

CONVERSATION FLOW:
1. After the user's query, if the dialog state is empty begin to ask questions to fill the dialog state
2. Use the appropriate search tools given the information contained in the dialog state to find matching restaurants
3. Ask the user for their desired location if not provided
4. If more than 5 restaurants match, ask clarifying questions to narrow down options
5. Once 5 or fewer restaurants remain, present them to the user with details
6. Be conversational and helpful throughout
7. If a reservation is created reset the dialog state by setting all values to "-"


When presenting restaurants, include:
- Restaurant name
- Location (city, state)
- Cuisine/category
- Key attributes
- Brief review summary
- Hours of operation

Always update the dialog state as you learn more about user preferences. Use "-" to indicate unknown preference for a particular dialogue state field. Do not assume values in the dialogue state that were not indicated by the user. For example, if the user mentioned a city name without specifying the state, elicit the state from the user.

You will be provided with input in the form below:
Dialog State: (dialog state)

User: (user's response)
"""

initial_answer = """\
Hello! I'm your restaurant reservation assistant. I can help you find the perfect place to dine. Just tell me what you're looking for - cuisine type, location, atmosphere, or any specific preferences!
"""

# Function declarations for Gemini (tool descriptions).
tools_dict = [ 
    {
                "name": "vdb_search_restaurants",
                "description": "Searches vector database for restaurants based on the query refined by the dialog state. Always create the vector database for the location before getting the restaurants",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for restaurants, it should contain information from the dialog state"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of top results to return",
                            "default": 100
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "update_dialog_state",
                "description": "Updates the dialog state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "City": {
                            "type": "string",
                            "description": "City" 
                        },
                        "State": {
                            "type": "string",
                            "description": "State, must be 2 letter abbreviation" 
                        },
                        "Cuisine": {
                            "type": "string",
                            "description": "Cuisine type (can be multiple)" 
                        },
                        "Name": {
                            "type": "string",
                            "description": "Name of restaurant (can be multiple)" 
                        },
                        "Misc": {
                            "type": "string",
                            "description": "Any additional constraints from the user" 
                        },
                        
                    },
                    "required": []
                }
            },
            {
                "name": "create_vdb_for_location",
                "description": "This creates a smaller vector database search only over restaurants at a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "City": {
                            "type": "string",
                            "description": "City" 
                        },
                        "State": {
                            "type": "string",
                            "description": "State, must be 2 letter abbreviation" 
                        }
                    },
                    "required": ["City", "State"]
                }
            },
            {
                "name": "make_reservation",
                "description": "Makes a reservation at a restaurant. Only call this function when the user explicitly confirms they want to make a reservation. The restaurant_id must be from the list of potential restaurants, and the date and time must be validated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restaurant_id": {
                            "type": "string",
                            "description": "The ID of the restaurant to make a reservation at. Must be from the list of potential restaurants."
                        },
                        "date": {
                            "type": "string",
                            "description": "The date for the reservation in YYYY-MM-DD format (e.g., 2025-10-15). Must be a future date."
                        },
                        "time": {
                            "type": "string",
                            "description": "The time for the reservation in HH:MM format using 24-hour notation (e.g., 19:30 for 7:30 PM)."
                        }
                    },
                    "required": ["restaurant_id", "date", "time"]
                }
            }]


def _normalize_tool_result(data):
    """Convert FastMCP client results into plain dictionaries when possible."""
    if isinstance(data, dict):
        return data
    if hasattr(data, "data") and isinstance(data.data, dict):
        return data.data
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return {"status": "success", "data": data}


async def execute_function_call(dialog_state, function_call):
    """
    Execute function calls requested by the Gemini model and return the results.
    
    This async function acts as a router that executes different function calls based on
    the function name provided by the Gemini model. It handles two execution paths:
    1. Local execution for dialog state updates
    2. Remote execution via MCP client for restaurant searches, vDB operations, and reservations

    Parameters
    ----------
    dialog_state : DialogState
        Current dialog state object tracking user preferences (City, State, Cuisine, Name, Misc)
    function_call : FunctionCall
        Gemini function call object containing:
            - name (str): Function name to execute
            - args (dict): Function arguments as key-value pairs
    
    Returns
    -------
    dict
        Result dictionary from the executed function. Structure varies by function:
        - For 'update_dialog_state': return {"status": "success"}
        - For MCP functions,: return data.data from the MCP client call, which includes:
            - {"status": "success", "restaurants": [...]} for vdb_search_restaurants
            - {"status": "success"} and possibly other required information for vDB operations, and reservations

    Execution Flow
    --------------
    1. Extract function name and arguments from function_call

    2. If function is 'update_dialog_state':
       - Call local update_dialog_state() with dialog_state and unpacked args. Hint: use '**function_args'.
       - Return success status

    3. For all other functions:
       - Establish async MCP client
       - Add dialog_state to args if function is 'vdb_search_restaurants'
       - Call the tool asynchronously via MCP client with function name and args
       - Return the tool execution result as specified above.
    
    """

    # 1. Extract function name and arguments from function_call
    function_name = function_call.name
    function_args = {key: val for key, val in function_call.args.items()}
    
    # 2. If function is 'update_dialog_state': ...
    if function_name == 'update_dialog_state':
        return update_dialog_state(dialog_state, **function_args)

    # 3. For all other functions: ...
    async with mcp_client:
        if function_name == 'vdb_search_restaurants':
            function_args['dialog_state'] = dialog_state.get_state()

        data = await mcp_client.call_tool(function_name, function_args)
        return _normalize_tool_result(data)







########## End of execute_function_call() #############


class RestaurantChatbot:
    def __init__(self):
        """
        Initialize the restaurant chatbot with Gemini AI configuration and conversation history.
        
        This constructor sets up:
        1. Gemini tool configuration with function declarations for restaurant operations
        2. Chat session with gemini-2.5-pro model
        3. Initial conversation history with system instructions
        4. Empty potential restaurants set
        5. DialogState object for tracking user preferences
        
        Attributes
        ----------
        config : types.GenerateContentConfig
            Gemini configuration with tool declarations
        potential_restaurants : set
            Set of restaurant indices matching user criteria
        chat : Chat
            Gemini chat session object with conversation history
        dialog_state : DialogState
            Object tracking user preferences (Location, Cuisine, Name, Misc)
        
        Examples
        --------
        >>> chatbot = RestaurantChatbot()
        >>> response = chatbot.chat_turn("I want Italian food in NYC")
        """
        tools = types.Tool(function_declarations = tools_dict)
        self.config = types.GenerateContentConfig(tools = [tools])
        self.potential_restaurants = set([])

        self.dialog_state = DialogState()
        self.last_restaurant_strs = []
        self.selected_restaurant_id = None

        self.chat = None
        if client is not None:
            self.chat = client.chats.create(
                model = MODEL_NAME,
                config = self.config,
                history = [
                    types.Content(role = "user",
                                  parts = [types.Part(text = initial_prompt)]),
                    types.Content(role = 'model',
                                  parts = [types.Part(text = initial_answer)])])

    def _extract_state_alias(self, text: str):
        lowered = text.lower()
        stripped = text.strip()
        abbreviation_match = re.search(r"(?:,\s*|\s+)(AL|FL|IN|NJ|PA|TN)\b", text)
        if not abbreviation_match and stripped in {"AL", "FL", "IN", "NJ", "PA", "TN"}:
            abbreviation_match = re.match(r"(AL|FL|IN|NJ|PA|TN)", stripped)
        if abbreviation_match:
            return abbreviation_match.group(1)

        for alias, abbr in STATE_ALIASES.items():
            if len(alias) == 2:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return abbr
        return None

    def _extract_city(self, text: str):
        patterns = [
            r"\bin\s+([A-Z][A-Za-z .'-]+?)(?:,|\s+(?:AL|FL|IN|NJ|PA|TN)\b|\s+(?:Alabama|Florida|Indiana|New Jersey|Pennsylvania|Tennessee)\b|$)",
            r"\bnear\s+([A-Z][A-Za-z .'-]+?)(?:,|\s+(?:AL|FL|IN|NJ|PA|TN)\b|\s+(?:Alabama|Florida|Indiana|New Jersey|Pennsylvania|Tennessee)\b|$)",
            r"^\s*([A-Z][A-Za-z .'-]+?)\s*,\s*(?:AL|FL|IN|NJ|PA|TN|Alabama|Florida|Indiana|New Jersey|Pennsylvania|Tennessee)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                city = match.group(1).strip(" .,")
                stop_words = {"a", "an", "the", "restaurant", "restaurants"}
                words = [word for word in city.split() if word.lower() not in stop_words]
                if words:
                    return " ".join(words)
        return None

    def _extract_cuisine(self, text: str):
        lowered = text.lower()
        for cuisine in KNOWN_CUISINES:
            if re.search(rf"\b{re.escape(cuisine)}\b", lowered):
                return "Sushi" if cuisine == "sushi" else cuisine.title()
        return None

    def _extract_misc(self, text: str):
        lowered = text.lower()
        misc = []
        if "dog" in lowered or "pet" in lowered:
            misc.append("dog-friendly")
        if "patio" in lowered:
            misc.append("patio")
        if "outdoor" in lowered:
            misc.append("outdoor seating")
        if "casual" in lowered:
            misc.append("casual")
        if "good" in lowered and "sushi" in lowered:
            misc.append("good sushi")
        return ", ".join(dict.fromkeys(misc)) or None

    def _update_state_from_user(self, user_message: str):
        lowered = user_message.lower()
        updates = {}

        state = self._extract_state_alias(user_message)
        city = self._extract_city(user_message)
        cuisine = self._extract_cuisine(user_message)
        misc = self._extract_misc(user_message)

        if city:
            updates["City"] = city
        if state:
            updates["State"] = state
        if cuisine:
            updates["Cuisine"] = cuisine
            if "changed my mind" in lowered or "instead" in lowered:
                updates["Name"] = "-"
                updates["Misc"] = "-"
        if misc:
            if self.dialog_state.Misc not in ("", "-"):
                existing = [item.strip() for item in self.dialog_state.Misc.split(",")]
                new_items = [item.strip() for item in misc.split(",")]
                updates["Misc"] = ", ".join(dict.fromkeys(existing + new_items))
            else:
                updates["Misc"] = misc

        if updates:
            update_dialog_state(self.dialog_state, **updates)

    def _missing_location_message(self):
        if self.dialog_state.City in ("", "-") and self.dialog_state.State in ("", "-"):
            return "What city and state should I search in?"
        if self.dialog_state.State in ("", "-"):
            return f"I have {self.dialog_state.City}. Which state is that in?"
        if self.dialog_state.City in ("", "-"):
            return "Which city should I search in?"
        return None

    def _preference_summary(self):
        parts = []
        if self.dialog_state.Cuisine not in ("", "-"):
            parts.append(self.dialog_state.Cuisine)
        if self.dialog_state.City not in ("", "-") and self.dialog_state.State not in ("", "-"):
            parts.append(f"in {self.dialog_state.City}, {self.dialog_state.State}")
        if self.dialog_state.Misc not in ("", "-"):
            parts.append(f"with {self.dialog_state.Misc}")
        return " ".join(parts) if parts else "restaurants"

    def _parse_restaurant_choice(self, user_message: str):
        lowered = user_message.lower()
        if not self.last_restaurant_strs:
            return None
        if "first" in lowered or re.search(r"\b1(st)?\b", lowered):
            return 0
        if "second" in lowered or re.search(r"\b2(nd)?\b", lowered):
            return 1
        if "third" in lowered or re.search(r"\b3(rd)?\b", lowered):
            return 2
        return None

    def _parse_reservation_datetime(self, user_message: str):
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        lowered = user_message.lower()
        date_match = re.search(
            r"\b(" + "|".join(month_names) + r")\s+(\d{1,2})\b",
            lowered,
        )
        if not date_match:
            return None, None
        time_text = lowered[date_match.end():]
        time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", time_text)
        if not time_match:
            time_matches = list(re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", time_text))
            time_match = time_matches[0] if time_matches else None
        if not time_match:
            return None, None

        month = month_names[date_match.group(1)]
        day = int(date_match.group(2))
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or "0")
        suffix = time_match.group(3)
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        return f"2025-{month:02d}-{day:02d}", f"{hour:02d}:{minute:02d}"

    async def _call_mcp_tool(self, name: str, args: dict):
        async with mcp_client:
            result = await mcp_client.call_tool(name, args)
            return _normalize_tool_result(result)

    async def _search_local_restaurants(self):
        await self._call_mcp_tool(
            "create_vdb_for_location",
            {"City": self.dialog_state.City, "State": self.dialog_state.State},
        )
        search_state = self.dialog_state.get_state()
        result = await self._call_mcp_tool(
            "vdb_search_restaurants",
            {
                "dialog_state": search_state,
                "query": self._preference_summary(),
                "k": 100,
            },
        )
        restaurants = result.get("restaurants", [])
        self.potential_restaurants = set(restaurants)
        if not restaurants:
            self.last_restaurant_strs = []
            return []

        restaurant_data = await self._call_mcp_tool(
            "get_restaurant_strs",
            {"restaurant_indexes": restaurants[:5]},
        )
        self.last_restaurant_strs = restaurant_data.get("restaurant_strs", [])
        return self.last_restaurant_strs

    def _format_restaurant_results(self, restaurant_strs):
        lines = [f"I found these matches for {self._preference_summary()}:"]
        for idx, restaurant in enumerate(restaurant_strs, start=1):
            lines.append(f"{idx}. {restaurant}")
        lines.append("Which one would you like, or would you like to narrow the search?")
        return "\n".join(lines)

    async def _fallback_chat_turn(self, user_message: str):
        self._update_state_from_user(user_message)

        choice = self._parse_restaurant_choice(user_message)
        if choice is not None:
            if choice >= len(self.last_restaurant_strs):
                return "I do not have that many restaurant options yet. Which listed restaurant would you like?"
            restaurant = self.last_restaurant_strs[choice]
            match = re.search(r"RestaurantId:\s*(\S+)", restaurant)
            self.selected_restaurant_id = match.group(1) if match else None
            date, time = self._parse_reservation_datetime(user_message)
            if not date or not time:
                name_match = re.search(r"Name:\s*(.*?)\s+Location:", restaurant)
                name = name_match.group(1) if name_match else "that restaurant"
                return f"Great, I can reserve {name}. What date and time would you like?"

        date, time = self._parse_reservation_datetime(user_message)
        if date and time and self.selected_restaurant_id:
            result = await self._call_mcp_tool(
                "make_reservation",
                {
                    "restaurant_id": self.selected_restaurant_id,
                    "date": date,
                    "time": time,
                },
            )
            if result.get("status") == "success":
                update_dialog_state(
                    self.dialog_state,
                    City="-", State="-", Cuisine="-", Name="-", Misc="-",
                )
                self.selected_restaurant_id = None
                self.last_restaurant_strs = []
                self.potential_restaurants = set()
                return f"Your reservation is confirmed for {date} at {time}."
            return "I could not make that reservation. Please check the restaurant, date, and time."

        missing_location = self._missing_location_message()
        if missing_location:
            if self.dialog_state.Cuisine not in ("", "-"):
                return f"I captured {self.dialog_state.Cuisine}. {missing_location}"
            return missing_location

        if self.dialog_state.Cuisine in ("", "-"):
            if self.dialog_state.Misc not in ("", "-"):
                return f"I captured {self.dialog_state.City}, {self.dialog_state.State}. What cuisine would you like?"
            return f"I captured {self.dialog_state.City}, {self.dialog_state.State}. What cuisine should I search for?"

        restaurant_strs = await self._search_local_restaurants()
        if not restaurant_strs:
            return f"I could not find matching restaurants for {self._preference_summary()}. Would you like to loosen one preference?"
        return self._format_restaurant_results(restaurant_strs)
    
    
    async def chat_turn(self, user_message: str, print_ds = False, verbose: bool = False) -> str:
        """
        Process one conversation turn with the user.
        
        This method handles a complete interaction cycle:
        1. Constructs full message with dialog state context
        2. Sends message to Gemini model
        3. Executes any function calls returned by the model, and send function call results to Gemini.
        4. Repeat 3, until model returns no function call
        5. Returns final assistant response
        
        Parameters
        ----------
        user_message : str
            User's input message/query
        print_ds : bool, optional
            If True, prints dialog state after response (default: False)
        verbose : bool, optional
            If True, prints debug information including:
            - User message with separators
            - Model response parts
            - Function calls with arguments
            - Function execution results
            - Dialog state (default: False)
        
        Returns
        -------
        str
            The assistant's text response to the user
        
        Behavior
        ---------------
        1. Append dialog state and reminder to user message
        2. Send enhanced message to Gemini
        3. While response contains function calls:
           a. Collect all function calls from response
           b. Execute each function via execute_function_call()
           c. Update potential_restaurants if function returns updated set
           d. Send function results back to model as function responses
        4. Extract and return final text response
        5. Print dialog state if print_ds or verbose is True
        
        Examples
        --------
        >>> chatbot = RestaurantChatbot()
        >>> response = chatbot.chat_turn("Find me pizza places", verbose=False)
        >>> print(response)
        "I'd be happy to help! What city and state are you looking in?"
        
        >>> response = chatbot.chat_turn("New York, NY")
        >>> print(response)
        "Great! I found several pizza places in New York. Here are the top options..."
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"USER: {user_message}")
            print(f"{'='*60}\n")

        if self.chat is None:
            assistant_message = await self._fallback_chat_turn(user_message)
            if print_ds or verbose:
                print(f"\nDialog State:\n{self.dialog_state.to_json()}\n")
            return assistant_message

        # Add system instruction context
        full_message = f"Dialog State:\n{self.dialog_state.to_json()}\n" + \
            f"Reminder if the user changes their mind reset the dialog state and if the location changes set vdb search.\n\n" + \
            f"User: {user_message}\nALWAYS Assume the year is 2025."
        
        # Send message to model
        try:
            response = self.chat.send_message(full_message)
        except ClientError as exc:
            if getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                assistant_message = await self._fallback_chat_turn(user_message)
                if print_ds or verbose:
                    print(f"\nDialog State:\n{self.dialog_state.to_json()}\n")
                return assistant_message
            raise
        
        if verbose:
            print(response.candidates[0].content.parts)

        # Tool use logic at this chat turn.
        while any(hasattr(part, 'function_call') and part.function_call for part in response.candidates[0].content.parts):
            function_calls = []
            
            # Collect all function calls
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
            
            # Execute function calls
            function_responses = []
            for fc in function_calls:
                if verbose:
                    print(f"Calling function: {fc.name}")
                    print(f"Arguments: {dict(fc.args)}")
                
                # This is were we call the tool correspondign to 'fc'.
                # This can can be any of the 4 tools that we described to Gemini in tools_dict.
                result = await execute_function_call(self.dialog_state, fc)
                
                if verbose:
                    print(f"Result: {result}\n")
                
                potential_restaurants_str = ''

                if fc.name == 'vdb_search_restaurants' and 'restaurants' in result:
                    self.potential_restaurants = set(result['restaurants'])
                    if not self.potential_restaurants:
                        potential_restaurants_str = "\nNo matching restaurants were found."
                    else:
                        async with mcp_client:
                            restaurant_data = await mcp_client.call_tool(
                                "get_restaurant_strs",
                                {"restaurant_indexes": list(self.potential_restaurants)},
                            )
                            restaurant_data = _normalize_tool_result(restaurant_data)
                            restaurant_strs = restaurant_data.get("restaurant_strs", [])
                            if restaurant_strs:
                                potential_restaurants_str = "\n" + "\n".join(restaurant_strs)
                        
                
                function_responses.append(
                    types.Part.from_function_response(
                        name = fc.name,
                        response = {"result": result['status'] + potential_restaurants_str},
                    ))
                
                if verbose:
                    print(function_responses)
            
            # Send function results back to model
            try:
                response = self.chat.send_message(function_responses)
            except ClientError as exc:
                if getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                    assistant_message = await self._fallback_chat_turn(user_message)
                    if print_ds or verbose:
                        print(f"\nDialog State:\n{self.dialog_state.to_json()}\n")
                    return assistant_message
                raise
            
            if verbose:
                print(response.candidates[0].content.parts)
        
        # Extract text response
        assistant_message = response.text
        
        if print_ds or verbose:
            print(f"\nDialog State:\n{self.dialog_state.to_json()}\n")
        
        return assistant_message


def main():
    """
    Main function to run the chatbot in CLI mode
    
    Behavior
    --------
    1. Creates RestaurantChatbot instance
    2. Displays welcome message and instructions
    3. Enters interactive loop:
       - Prompts for user input
       - Exits on 'quit', 'exit', or 'bye' commands
       - Skips empty input
       - Processes input with chat_turn(verbose=True) for debug output
       - Displays assistant response with formatting
    4. Prints goodbye message on exit
    
    Exit Commands
    -------------
    - 'quit'
    - 'exit'
    - 'bye'
    (case-insensitive)

    """
    chatbot = RestaurantChatbot()
    
    print("Restaurant Reservation Chatbot")
    print("=" * 60)
    print("Ask me to help you find a restaurant!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using the Restaurant Reservation Chatbot!")
            break
        
        if not user_input:
            continue
        
        # Use verbose = True for CLI mode to see debug output
        response = asyncio.run(chatbot.chat_turn(user_input, verbose = True))
        print(f"\nAssistant: {response}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
