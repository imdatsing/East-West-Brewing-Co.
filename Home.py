import streamlit as st

def app():
    
    with st.container():      
        col1, col2 = st.columns([4,3])
        
        with col1:
            st.header("East West Brewing Co")
        
        with col2:
            st.image("Images/1.jpg")
            
        with col1: 
            st.write("East West Brewing Co is **:orange[Vietnam's first craft brewery brand]**, with **:orange[two branches in Ho Chi Minh City and Da Nang]**. The company was founded with the vision of offering **:orange[world-class craft beer]** to the Vietnamese community.")
            st.write("")
            st.write("The brewery produces a diverse range of beers, including both **:orange[core and premium varieties]**, combining **:orange[global and local ingredients]**. East West Brewing Co. takes pride in connecting cultures through their craft beer and invites everyone to experience the **:orange[unique offerings of the East West brand]**.")
         
    with st.container():
        st.write("___")
        
        col1, col2 = st.columns([4,3])
        
        with col1: 
            st.header("Why East West Brewing Co need us?")
            
        with col2: 
            st.image("Images/2.jpg")
            
        with col1:
            st.write("The brand intends to expand its presence to additional coastal regions of Vietnam. To facilitate this expansion, **:orange[East West Brewing Co aims to enhance service quality at its Da Nang facility]**, setting a benchmark for future locations. Through rigorous analysis of operational data across departments and utilizing advanced data science techniques, the company seeks to understand the correlation between customer experience and monthly revenue.")
            st.write("")
            st.write("However, the Research and Development department currently lacks an in-house data specialist for making data-driven decisions. As a result, **:orange[East West Brewing Co has decided to enlist the services of an external agency]**. This agency will conduct comprehensive research, offer insightful analyses and recommendations, and develop data science solutions to help achieve the brand's objectives.")
            
    with st.container():
        st.write("___")
        
        col1, col2 = st.columns([4,3])
        
        with col1:
            st.header("What we offer East West Brewing?")
        
        with col2:
            st.image("Images/3.jpg")
            
        with col1:
            st.subheader("1️⃣ Model 1: Customer Traffic Prediction")
            st.write("  -   First function: Predict the customer traffic for **:orange[the next year (2025)]** to support for sale marketing strategy.")
            st.write("  -   Second function: Predict the customer traffic for **:orange[the next 5 days]** to support for operations as staffing and preparing cooking ingredients.")
            st.write((""))
            
            st.subheader("2️⃣ Model 2: AI Chatbot")
            st.write("  - The chatbot can **:orange[recommend dishes]** based on user preferences and dietary needs, **:orange[provide detailed menu information]** and **:orange[answer questions about the restaurant]**'s location, contact details, opening hours, etc.")
            st.write((""))
            
            st.subheader("3️⃣ Model 3: Customer Feedback Analysis")
            st.write("  - First function: Label the feedback given as: food; service; atmosphere; food/service; food/atmosphere; service/atmosphere; food/service/atmosphere.")
            st.write("  - Second function: Sentiment the feedback given to see if their overall experience is good (positive)/ nothing much (neutral) of bad (negative).")
            
    st.image("Images/4.jpg")
