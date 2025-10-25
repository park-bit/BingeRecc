# BingeRecc

a simple movie recommendation app made with streamlit + python.  
basically you search a movie and it gives you similar ones. that’s it.  

---

## What this does
- search any movie and get recommendations based on that
- has a basic clean ui (streamlit)
- uses csv files (metadata, ratings etc)
- lightweight, no heavy backend nonsense
- you can probably add more data or logic if you’re not lazy

---

## Tech used
- python
- jupyter notebook (i just wanted to play around with those csvs innit)
- streamlit
- pandas / numpy
- sklearn (for similarity stuff)
- some tmdb dataset files
- tmdb and obdb api keys

---

## Files
- app.py # main app
- requirements.txt # yeah install these
- movies_metadata.csv # movie info
- ratings.csv # user ratings (useless, u can add this if u want)
- credits.csv # cast + crew
- keywords.csv # keywords lol
- app_icon.ico # icon for the app

theres also the python notbeook for the initial application, u can use it to clean the dataset further if u want and also add sum features iyw.....


---

## Screenshots

<img width="1920" height="1080" alt="Screenshot (665)" src="https://github.com/user-attachments/assets/1d516d02-e133-4a7c-956e-d14a41f5812c" />

<img width="1920" height="1080" alt="Screenshot (667)" src="https://github.com/user-attachments/assets/720c63bc-294c-4ca3-83c6-7eb2103c32d9" />

<img width="1920" height="1080" alt="Screenshot (669)" src="https://github.com/user-attachments/assets/39bbd0e2-2eaf-4c19-b058-7bb63ca939e6" />

<img width="1920" height="1080" alt="Screenshot (671)" src="https://github.com/user-attachments/assets/e3dcadad-bfc3-4a63-9d95-1c1da56062b8" />



---

## How to run
clone the repo  
make a venv if you want  
install requirements  
then just run streamlit app


open the localhost link streamlit gives you and it should just work (it'll open automatically sumtimes)
if it doesn’t, you probably missed something obvious u nerd

---

## how it works
it takes your selected movie, checks for similar stuff using keywords / genres / ratings,  
and spits out a list of recommendations
no rocket science, just vectorizing and cosine similarity stuff behind the scenes (and also sneaky use of api keys)

---

## Customize it
you can:
- make it coooooler
- add login or save features
- use a better algo if you actually care (I certainly don't)
- deploy it on streamlit cloud or whatever

---

## Why i made this
wanted something quick that shows how rec systems work (kidding i just wanted to fk around and find out)
plus i was bored (yeah i was)
and streamlit makes ui easy, so yeah (and also free cloud deployment)

---

## License
MIT. do whatever you want with it....

---

peace out
hare krishna.
