import streamlit as st                       # to make the user interface of the app
import pickle                                # to load model and tfidf vectorizer
import re                                    # to filter out punctuations and numbers from text
from nltk.tokenize import word_tokenize      # to tokenize text
from nltk.corpus import stopwords            # to remove stopwords
import time                                  # for animations


def prediction(tfidf_text):
    '''
    This function uses tfidf vector to predict the ticket-type. 
    And returns the prediction as well as prediction probabilty too.
    '''
       
    # loading model
    model = pickle.load(open('Ticket_Classification_Model.pkl','rb'))
    
    # predicting ticket-type
    pred = model.predict(tfidf_text)[0]
    # prediction probability
    pred_prob = round(max(model.predict_proba(tfidf_text)[0])*100,2)
    
    with st.spinner(text = 'Predicting...'):
        time.sleep(3)
    
    # returning prediction with its probability
    return pred, pred_prob




def preprocess_text(text):
    '''
    This function allows to convert the text data into tf-idf vector, remove the stopwords, one last checker and then returns it.
    '''
    with st.spinner(text = 'Analyzing ticket information...'):
        time.sleep(3)
    
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word.strip() for word in word_tokenize(str(re.sub('[^A-Za-z]+', ' ', text.lower()))) if word not in stop_words)
    
    # checker
    if clean_text=='':
        st.error('The information you have entered, does not contain any appropriate knowledge. Kindly reframe your query.')
        st.stop()
    
    # loading already fitted tfidf vectoriser
    tfidf = pickle.load(open('tfidf.pkl','rb'))
    
    # transforming text
    tfidf_text = tfidf.transform([clean_text])
    
    # returning tfidf text vector
    return tfidf_text.todense()


def main():
    # app title
    st.header('Customer Ticket Classifier')
    
    html_temp = '''
    <div style="background-color:tomato; padding:20px; border-radius: 25px;">
    <h2 style="color:white; text-align:center; font-size: 30px;"><b>Customer Support Ticket Classification Model</b></h2>
    </div><br><br>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # text input
    text = st.text_area('Please input ticket information:')
       
    # predicting ticket_type
    if st.button('Predict'):
        
        # necessary requirements
        
        # checker for empty text
        if text.strip()=='':
            st.warning('No information has been written! Kindly write your query again.')
            st.stop()
            
        # checker for punctuation only in the text
        if str(re.sub(r"[~`!@#$%^*()[]{}_-+=;'\,./:']+", ' ', text)).strip()=='':
            st.warning('You have written punctuation only. Kindly write a proper query again.')
            st.stop()
            
        # checker for numbers only in the text
        if str(re.sub(r"[0-9]+", ' ', text)).strip()=='':
            st.warning('You have written numbers only. Kindly write a proper query again.')
            st.stop()
        
        # text should have atleast 5 words in it.
        if len(text.split(' ')) < 5:
            st.warning('Ticket information provided is too low. Kindly write atleast five words in the query.')
            st.stop()
            
        # preprocessing of text
        tfidf_text = preprocess_text(text)
            
        # predicting ticket-type
        pred, pred_prob = prediction(tfidf_text)
                
        # result display
        ticket_type = {0 : 'a Bot. ', 1 : 'an Agent. '}
        result = 'The given user query will be resolve by ' + ticket_type[pred]
        acc = 'The model is ' + str(pred_prob) +'% confident about it.' 
        st.success(result + '\n' +acc)
        
    
if __name__=='__main__':
    main()