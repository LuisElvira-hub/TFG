#-----------------------------------------------------------------------------------------------------------------------
#Zona de imports
#-----------------------------------------------------------------------------------------------------------------------
#MAIN
import streamlit as st
import joblib
import re
#detectar lenguaje
from langdetect import detect
#buscador de noticas
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk #lo usamos para las stop words, que funcione el programa en ingles y español
from nltk.corpus import stopwords
from requests.utils import quote #para las busquedas en url
nltk.download('stopwords')
stopwords_ingles=stopwords.words('english')
stopwords_espanol=stopwords.words('spanish')
stopwords_combinadas=stopwords_ingles+stopwords_espanol
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#-----------------------------------------------------------------------------------------------------------------------
#funcionalidad de detectar lenguaje
#-----------------------------------------------------------------------------------------------------------------------
def detectar_idioma(texto):
    try:
        return detect(texto)
    except:
        return "error"


#-----------------------------------------------------------------------------------------------------------------------
#funcionalidad de buscar en medios
#-----------------------------------------------------------------------------------------------------------------------
#creamos un extractor de palabras clave para que podamos buscar las noticias
def extractor_palabras_clave(texto, num=3):
    Vectorizador_2= CountVectorizer(stop_words=stopwords_combinadas)
    X=Vectorizador_2.fit_transform([texto])
    suma_palabras=X.toarray().sum(axis=0)
    palabras=Vectorizador_2.get_feature_names_out()
    ranking= [(palabras[i], suma_palabras[i]) for i in range(len(palabras))]
    ranking.sort(key=lambda x: x[1], reverse=True)
    return [palabra for palabra, _ in ranking[:num]]

#Creamos el url de busqueda por que cada medio tiene una estructura de url dif
def construir_url(medio, palabras_clave):
    if medio=="eldiario": #izquierdas español
        busqueda=" ".join(palabras_clave)
        return f"https://www.eldiario.es/busqueda/{quote(busqueda)}"
    elif medio=="elconfidencial": #derechas español
        #limito longitud a 20
        max_len=20
        busqueda_final=""
        for palabra in palabras_clave:
            if busqueda_final:
                nueva=busqueda_final+ "-"+palabra
            else:
                nueva=palabra
            if len(nueva)<=max_len:
                busqueda_final=nueva
            else:
                break
        return f"https://www.elconfidencial.com/buscar/2-6-1-3/0/1/10/desc/{busqueda_final}/"
    elif medio=="jacobin":
        busqueda=" ".join(palabras_clave)
        return f'https://jacobin.com/search?query={quote(busqueda)}'
    elif medio=="nypost":
        busqueda="+".join(palabras_clave)
        return f"https://nypost.com/search/{quote(busqueda)}/"
    elif medio=="maldita":
        busqueda=" ".join(palabras_clave)
        return f"https://maldita.es/buscar/{quote(busqueda)}/"
    elif medio=="dispatch":
        busqueda=" ".join(palabras_clave)
        return f"https://thedispatch.com/category/fact-check/?sort=&term={quote(busqueda)}"
    
    else:
        return "#"
    
    
#Creamos una def que pueda darnos titulares mas recientes, va a variar dependiendo de la pag web que queramos acceder-----------------

def obtener_titulares(medio, url, max_titulares=3):
    titulares=[]
    try:
        headers={"User-Agent": "Mozilla/5.0"}#usamos este agente por que el otro bloqueaba mis scrappings ;)
        respuesta=requests.get(url, headers=headers, timeout=10)
        soup=BeautifulSoup(respuesta.content, "html.parser")

        if medio=="eldiario":
            articulos=soup.find_all("li", class_="article-cont-search")[:max_titulares]
            for art in articulos:
                div=art.find("div", class_="second-column")
                if div:
                    a=div.find("a")
                    if a and a.text.strip():
                        titulo=a.text.strip()
                        link=a["href"]
                    titulares.append((titulo, link))
        elif medio=="elconfidencial":
            articulos=soup.find_all("h2", class_="new-title")[:max_titulares]
            for art in articulos:
                a=art.find("a")
                if a and a.text.strip():
                    titulo=a.text.strip()
                    link=a["href"]
                titulares.append((titulo, link))
        elif medio=="jacobin":
            articulos=soup.find_all("h1", class_="ar-mn__title")[:max_titulares]
            for art in articulos:
                a=art.find("a")
                if a and a.text.strip():
                    titulo=a.text.strip()
                    link=a["href"]
                titulares.append((titulo, link))
        elif medio=="nypost":
            articulos=soup.find_all("h3", class_="story__headline headline headline--archive")[:max_titulares]
            for art in articulos:
                a=art.find("a")
                if a and a.text.strip():
                    titulo=a.text.strip()
                    link=a["href"]
                titulares.append((titulo, link))
        elif medio=="maldita":
            articulos=soup.find_all("div", class_="flex flex-col")[:max_titulares]
            for art in articulos:
                a=art.find("a")
                if a and a.text.strip():
                    titulo=a.text.strip()
                    link=a["href"]
                titulares.append((titulo, link))
        elif medio=="dispatch":
            articulos=soup.find_all("div", class_="flex items-start gap-2")[:max_titulares]
            for art in articulos:
                a=art.find("a")
                if a and a.text.strip():
                    titulo=a.text.strip()
                    link=a["href"]
                titulares.append((titulo, link))
    except Exception as e:
        titulares.append((f"Error al obtener titulares de {medio}: {e}", "#"))
    return titulares
#-----------------------------------------------------------------------------------------------------------------------
#FUNCIONALIDAD DE DETECTAR SENTIMIENTOS
#-----------------------------------------------------------------------------------------------------------------------
#Creamos un programa para que tambien podamos detectar el sentimiento que te intenta hacer sentir la noticia
#HAY QUE ACORTARLO O DA ERRORES
tokenizer= AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
modelo=AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
analizador_sentimiento=pipeline("sentiment-analysis", model=modelo, tokenizer=tokenizer)
def detectar_sentimiento(texto):
    texto_corto=texto[:1000]
    resultado=analizador_sentimiento(texto_corto)[0]
    label=resultado['label']
    puntuacion=int(label.split()[0])

    if puntuacion >=4:
        sentimiento="Positivo"
    elif puntuacion ==3:
        sentimiento="Neutral"
    else:
        sentimiento="Negativo"

    return sentimiento, resultado['score']

#-----------------------------------------------------------------------------------------------------------------------
#MAIN
#-----------------------------------------------------------------------------------------------------------------------
#metemos el limpiador de antes
def limpiador(text):
    text=text.lower()
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"[^a-zA-ZñÑáéíóúÁÉÍÓÚüÜçÇ\s]","",text) #así limpiamos los caracteres que no vamos a poder leer bien
    text=re.sub(r"\s+"," ",text).strip()
    return text
#metemos el modelo y el vectorizador
modelo=joblib.load("scripts/modelos/modelofakenews1.pkl")
Vectorizador=joblib.load("scripts/modelos/vectorizadorfakenews1.pkl")
#creo la app
st.title("Detector de Fake News")
st.write("Introduce un articulo y te diré si es fake news")
#aqui el usuario introduce
if "input1" not in st.session_state:
   st.session_state.input1=""
input1=st.text_area("Introduce el articulo aquí:", st.session_state.input1)

#metemos botones
#col1,col2=st.columns(2) NO FUNCIONA

#predecimos
#with col1:
if st.button("Iniciar Analisis"):
    if input1.strip()=="":
        st.warning("Introduce texto por favor")
    else:
        input1_limpio=limpiador(input1)
        vector_input1=Vectorizador.transform([input1_limpio])
        prediccion1=modelo.predict(vector_input1)[0]
        probas=modelo.predict_proba(vector_input1)[0]
        #resultado
        label=""
        confianza=probas[prediccion1]*100
        if prediccion1==1:
            label="No es fake news"
        else:
            label="Fake News"
        st.subheader("Resultado:")
        st.markdown(f"**{label}**con**{confianza:.2f}%** de confianza.")
        #deteccion de sentimiento
        sentimiento, confianza_sentimiento=detectar_sentimiento(input1)
        st.subheader("Sentimiento de la noticia:")
        st.markdown(f"**{sentimiento}** con **{confianza_sentimiento*100:.2f}%** de confianza")
        #detectamos idioma
        try:
            idioma=detect(input1)
            idioma_detectado="Español" if idioma == "es" else "Inglés"
            st.markdown(f"Idioma detectado: **{idioma_detectado}**")
        except:
            st.error("No se detecta el idioma")
            idioma_detectado=None
        #Obtenemos palabras clave
        palabras_clave=extractor_palabras_clave(input1_limpio)
        st.markdown(f"Las palabras clave son: `{', '.join(palabras_clave)}`")
        #medios por ideologia
        if idioma_detectado=="Español":
            izquierda="eldiario"
            derecha="elconfidencial"
            organizacion="maldita"
        elif idioma_detectado=="Inglés":
            izquierda="jacobin"
            derecha="nypost"
            organizacion="dispatch"
        else:
            izquierda=derecha=None
        #mostramos enlaces (DEBUGGING)
        if izquierda and derecha:
            url_izq=construir_url(izquierda, palabras_clave)
            url_der=construir_url(derecha, palabras_clave)
            url_org=construir_url(organizacion, palabras_clave)
            titulares_izq=obtener_titulares(izquierda, url_izq)
            titulares_der=obtener_titulares(derecha, url_der)
            titulares_org=obtener_titulares(organizacion, url_org)
            st.subheader("Busqueda en medios segun ideología y en medios anti Bulos:")
            col1,col2,col3=st.columns(3)
            with col1:
                st.markdown(f"**Medio de Izquierdas**")
                st.markdown(f"[Ver busqueda en {izquierda}]({url_izq})", unsafe_allow_html=True)
                for titulo, link in titulares_izq:
                    st.markdown(f"-[{titulo}]({link})", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Medio de Derechas**")
                st.markdown(f"[Ver busqueda en {derecha}]({url_der})", unsafe_allow_html=True)
                for titulo, link in titulares_der:
                    st.markdown(f"-[{titulo}]({link})", unsafe_allow_html=True)    
            with col3:
                st.markdown(f"**Organización Anti Bulos**")
                st.markdown(f"[Ver busqueda en {organizacion}]({url_org})", unsafe_allow_html=True)
                for titulo, link in titulares_org:
                    st.markdown(f"-[{titulo}]({link})", unsafe_allow_html=True)   

#boton para borrar (no funciona)
#with col2:
   #if st.button("Borrar texto"):
    #st.session_state.input1=""
