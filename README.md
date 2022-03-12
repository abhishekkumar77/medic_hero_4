# medic_hero_4
A artificial intelligence based medical chatbot.

Project Setup
1. Install python 
2. Install **virtualenv** module using command **pip install virtualenv**
3. Create a virtual environment using command **python -m virtualenv project**
4. Activate the created virtual environment using command **project\Scripts\activate** Note: **project** is your environment name and you will notice (project) on your CLI
5. Install the required packages using command **pip install -r requirements.txt**
6. Setup Django by running command **python manage.py migrate** Note: Only run this command when you setting up Django for first time or when there change in Django configuration
7. Start the local Django server using command **python manage.py runserver 8080**
8. If bot is not responding, try to install some nltk modules by adding the below code in the **home\views.py** after line no.7
        **nltk.download('punkt')**
        **nltk.download('wordnet')**
        **nltk.download('omw-1.4')**
