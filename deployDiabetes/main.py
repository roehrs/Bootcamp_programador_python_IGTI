from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib


app = Flask(__name__)

def previsao_diabetes(lista_valores_formulario):
	prever= np.array(lista_valores_formulario).reshape(1,8) #transforma os valores do formulario
	modelo_salvo = joblib.load('melhor_modelo.sav') #realiza a carga do modelo salvo
	resultado = modelo_salvo.predict(prever) #aplica a previsao
	return resultado[0]

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/result',methods=['POST'])
def result():
	if request.method=='POST':
		lista_formulario=request.form.to_dict()
		lista_formulario=list(lista_formulario.values())
		lista_formulario=list(map(float, lista_formulario))
		resultado=previsao_diabetes(lista_formulario)
		if int(resultado)==1:
			previsao='Possui diabetes'
		else:
			previsao='Nao possui diabetes'

		#retorna o resultado para uma pagina HTML
		return render_template("resultado.html", previsao=previsao)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080, debug=True)