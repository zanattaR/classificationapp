import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import os
import pickle
import nltk
nltk.download('punkt')
import re
import string
import base64
from io import BytesIO
import unicodedata
import xlsxwriter


# Imagem
img = Image.open('logo_ps_class.png')
st.image(img)

# Título
st.markdown('## Modelo de Classificações Inglês e Espanhol')
st.write("")
#st.title('Inglês e Espanhol')
st.write('''Esta aplicação tem como objetivo de classificar em Categorização e Sentimentação, reviews em inglês e espanhol.
	Este MVP será usado para teste de viabilidade dos modelos preditivos.''')

# Funções

# Função para remover acentos
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: 
        pass
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return str(text)

# Função limpeza de texto
def clean_text_round_1(text):
    text = text.lower() # Letra minúscula
    text = strip_accents(text) # Removendo acentos das palavras
    
    remove = string.punctuation # deixando apenas os pontos de exclamação e interrogação
    remove = remove.replace('?', '')
    remove = remove.replace('!', '')
    remove = remove.replace('#', '')
    pattern = r"[{}]".format(remove) 
    text = re.sub(pattern, "", text) 
    
    text = re.sub('\w*\d\w*', '', text) # removendo digitos
    text = re.sub('\n', '', text) # Removendo quebras de linha
    return text

# Função para fazer predição - Categorização
def make_prediction_cat(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round_1)
	result = model_cat.predict(text)

	return result

# Função para fazer predição - Sentimentação
def make_prediction_sent(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round_1)
	result = model_sent.predict(text)

	return result

# Função para transformar df em excel
def to_excel(df):
	output = BytesIO()
	writer = pd.ExcelWriter(output, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='Planilha1',index=False)
	writer.save()
	processed_data = output.getvalue()
	return processed_data
	
# Função para gerar link de download
def get_table_download_link(df):
	val = to_excel(df)
	b64 = base64.b64encode(val)
	return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download</a>'

####### Upload dataset #######
#st.subheader('Dados')
data = st.file_uploader("Insira a base de dados", type='xlsx')
if data is not None:
	df = pd.read_excel(data, usecols=['DATE','TEXT'])
	df['DATE'] = df['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y').strftime('%d/%m/%Y'))
	st.write(df.head(50))
	st.write('Mostrando as 50 primeiras linhas da base de dados')
	st.write('Esta visualização já possui um pré-processamento')

# Selecionando o idioma
st.subheader('Selecione o idioma para a classificação')

idioma = st.selectbox('',('Inglês', 'Espanhol'))

if idioma == 'Inglês':

	# Carregando modelo de Categorização-Ing
	with open('model_cat_ing', 'rb') as f_cat:
		model_cat = pickle.load(f_cat)

	# Carregando modelo de Sentimentação-Ing	
	with open('model_sent_ing', 'rb') as f_sent:
		model_sent = pickle.load(f_sent)
else:

	# Carregando modelo de Categorização-Esp
	with open('model_cat_esp', 'rb') as f_cat:
		model_cat = pickle.load(f_cat)

	# Carregando modelo de Sentimentação-Esp
	with open('model_sent_esp', 'rb') as f_sent:
		model_sent = pickle.load(f_sent)


####### Fazendo previsões #######
st.subheader('Clique em Predict para obter as classificações da sua base de dados')
st.write('')

btn_predict = st.button('Predict')

if btn_predict:

	reviews = df['TEXT']

	# Aplicando modelo de Categorização e Sentimentação nos reviews
	reviews_cat = make_prediction_cat(reviews)
	reviews_sent = make_prediction_sent(reviews)

	# Criando dataframe com a data, review, categorização e sentimentação
	reviews_done = pd.DataFrame({'Data': df['DATE'],
		'Review': df['TEXT'],
		'Categorização': reviews_cat,
		'Sentimentação': reviews_sent})

	st.write(reviews_done.head(50))
	st.write('Clique em Download para baixar o arquivo')
	st.markdown(get_table_download_link(reviews_done), unsafe_allow_html=True)