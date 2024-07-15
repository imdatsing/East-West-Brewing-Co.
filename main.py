import streamlit as st
from streamlit_option_menu import option_menu

import Model1, Model2, Model3, Home

# Set page configuration
st.set_page_config(page_title="East West Brewing Co.", layout="wide")

# Main app class
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='FUNCTION MENU',
                options=['Home', 'Model 1', 'Model 2', 'Model 3'],
                icons=['house-fill', 'envelope-open-heart-fill', 'envelope-open-heart-fill', 'envelope-open-heart-fill'],
                menu_icon='cast',
                default_index=0
            )

        if app == 'Model 1':
            Model1.app()
        elif app == 'Model 2':
            Model2.app()
        elif app == 'Model 3':
            Model3.app()
        elif app == 'Home':
            Home.app()

app = MultiApp()
app.run()
