import streamlit as st
from openai import OpenAI
import json
import time

def app():
    st.title("Model 2: AI Chatbot")
    
    st.subheader("Xin chào! Tôi là **trợ lý ảo** của **East West Brewing Co** 	🤖, sẵn sàng giúp bạn với các gợi ý món ăn, thông tin về thực đơn và nhà hàng. Hãy đặt câu hỏi của bạn!	💓	💓	💓")

    client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

    # Initialize session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load the menu and restaurant infor once and store it in the session state
    if "menu" not in st.session_state:
        st.session_state.menu = load_json('menu.json')
    if "restaurant_info" not in st.session_state:
        st.session_state.restaurant_info = load_json('restaurant_info.json')
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask for a food recommendation or any menu-related question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Construct the system message with menu context
        menu_context = json.dumps(st.session_state.menu, indent=2)
        restaurant_info_context = json.dumps(st.session_state.restaurant_info, indent=2)
        system_message = {
            "role": "system",
            "content": f"You are a helpful restaurant food recommendation bot. You should only provide information related to the menu, food recommendations, and restaurant information. Here is the menu: {menu_context}. If the customer wants to see the menu, the menu will be displayed as a table with 6 columns: 3 columns for dishes and 3 columns for prices in VND. Here is the restaurant information: {restaurant_info_context}.  Avoid answer any questions outside these topics."
        }

        # Prepare the messages for the API request
        messages = [system_message] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=messages,
                stream=True
            )
            response = st.write_stream(stream)
            time.sleep(2)

        st.session_state.messages.append({"role": "assistant", "content": response})
    
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
