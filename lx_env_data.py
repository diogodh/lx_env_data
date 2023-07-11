#!/usr/bin/python
# -*- coding: utf-8

''' Example of the URL from Lisbon Ambiental Monitoring Parameters
https://dados.cm-lisboa.pt/dataset/monitorizacao-de-parametros-ambientais-da-cidade-de-lisboa/resource/2b537ede-4b62-4fb9-9404-d90b48cd9483
https://dados.cm-lisboa.pt/dataset/e7f31452-7d21-4b26-9eb5-1db62737a12d/resource/d8837f32-1f7e-4a61-bb8a-a2333f822edb/download/metadadosdadosabertos16082021.pdf
'''

import sys, os
import urllib.request, json 
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.system('cls')

date_start = input("Data início (AAAAMMDDHHMM): ")
print()
date_end = input("Data fim (AAAAMMDDHHMM): ")
print()

#Here are the location of the environmental sensors with the respective code 

loc = [
    ['0001 CALÇADA DA AJUDA', '0002 RUA GONÇALO VELHO CABRAL (EMQA)', '0003 CAIS DO SODRÉ'],
    ['0004 RUA DOS LUSIADAS (JFALCÂNTARA)', '0005 AV 24 JULHO', '0006 AVENIDA INFANTE SANTO'],
    ['0007 AVENIDA INFANTE DOM HENRIQUE (SANTA APOLÓNIA)', '0008 RUA DO OURO', '0009 PRAÇA DO COMÉRCIO'],
    ['0010 RUA SÁ NOGUEIRA (BOMBEIROS)', '0011 AVENIDA DE CEUTA (ETAR)', '0012 RUA DE SÃO BENTO (MERCADO)'],
    ['0013 COSTA CASTELO', '0014 PRAÇA DO MARTIM MONIZ', '0015 CAMPO SANTA CLARA'],
    ['0016 PRAÇA SÃO JOÃO BOSCO (CEMITERIO)','0017 RUA DA ESCOLA POLITÉCNICA (JARDIM BOTÂNICO)', '0018 ESTRADA DA CIRCUNVALAÇÃO (P. CAMPISMO)'],
    ['0019 ESTRADA DE MONTES CLAROS', '0020 AVENIDA DA LIBERDADE (EMQA)', '0021 RUA DOS SAPADORES'],
    ['0022 RUA MARIA PIA', '0023 AVENIDA ALMIRANTE REIS', '0024 RUA BRANCAMP'],
    ['0025 ESTRADA DO BARCAL (ESPAÇO BIODIVERSIDADE)', '0026 PARADA DO ALTO DE SÃO JOÃO (CEMITÉRIO)', '0027 ALAMEDA EDGAR CARDOSO'],
    ['0028 AVENIDA INFANTE DOM HENRIQUE (HUB BEATO)', '0029 AVENIDA FONTES PEREIRA MELO', '0030 AVENIDA ANTÓNIO AUGUSTO AGUIAR'],
    ['0031 LARGO MADRE DEUS', '0032 RUA CAMPOLIDE (ESCOLA)', '0033 RUA MORAIS SOARES'],
    ['0034 AVENIDA DA RÉPUBLICA', '0035 PRAÇA SÃO FRANCISCO ASSIS', '0036 ESTRADA DE MONSANTO'],
    ['0037 AVENIDA COLUMBANO BORDALO PINHEIRO', '0038 RUA PEDRO AZEVEDO', '0039 ESTRADA DE BENFICA'],
    ['0040 AVENIDA JOAO XXI', '0041 PARQUE RIBEIRINHO ORIENTE', '0042 TRAVESSA DE FRANCISCO REZENDE (QUINTA DA FONTE)'],
    ['0043 AVENIDA ALMIRANTE GAGO COUTINHO', '0044 AVENIDA SANTO CONDESTAVEL', '0045 RUA FREI CARLOS (ESCOLA)'],
    ['0046 CAMPO GRANDE (EMQA)', '0047 AVENIDA ESTADOS UNIDOS DA AMÉRICA', '0048 AVENIDA LUSÍADA'],
    ['0049 AVENIDA DE ROMA', '0050 RUA DOUTOR JOSÉ ESPÍRITO SANTO (BOMBEIROS)', '0051 AVENIDA JOSÉ REGIO (PARQUE DA BELA VISTA)'],
    ['0052 AVENIDA COLÉGIO MILITAR (QUINTA DA GRANJA)', '0053 RUA LÚCIO AZEVEDO', '0054 AVENIDA MARECHAL GOMES DA COSTA'],
    ['0055 AVENIDA BRASIL', '0056 AVENIDA GENERAL NORTON DE MATOS', '0057 CAMPO GRANDE (MUSEU DA CIDADE)'],
    ['0058 RUA PROFESSOR FRANCISCO GENTIL (JARDIM )', '0059 RUA QUINTA DA GRAÇA (PARQUE DA VINHA)', '0060 RUA CIDADE DE LOBITO (QUINTA PEDAGÓGICA)'],
    ['0061 AVENIDA MARIA HELENA VIEIRA SILVA', '0062 ESTRADA PACO LUMIAR', '0063 RUA DO RIO ZÊZERE (CEMITÉRIO)'],
    ['0064 ALAMEDA DA ENCARNACAO', '0065 AVENIDA DOUTOR ALFREDO BENSAUDE', '0066 RUA DA ILHA DOS AMORES (ESCOLA)'],
    ['0067 RUA VASCO DA GAMA FERNANDES (ESCOLA)', '0068 AVENIDA CIDADE DO PORTO (CML)', '0069 CALÇADA DA CARRICHE'],
    ['0070 RUA CHEN HE (ETAR)', '0071 ESTRADA MILITAR ÀS GALINHEIRAS (ESCOLA)', '0072 AVENIDA ULISSES (PARQUE CABEÇO ROLAS)'],
    ['0073 RUA ALFERES MALHEIRO (PARQUE JOSÉ GOMES FERREIRA)', '0074 RUA DA VENEZUELA', '0075 ALAMEDA PADRE ÁLVARO PROENÇA (EMQA)'],
    ['0076 AVENIDA DA LIBERDADE (RESTAURADORES)', '0077 TRAVESSA DOS INGLESINHOS (MERCADO)', '0078 RUA DE SÃO BERNARDO (JARDIM GUERRA JUNQUEIRO)'],
    ['0079 AVENIDA DOUTOR FRANCISCO LUÍS GOMES (EMQA)', '0080 RUA NAU CATRINETA', ''],
    ]

for row in loc:
    print("{: >20} {: >20} {: >20}".format(*row) + " | ")
    
print("\n\n")

local = input("Código Local (00xx): ")
print("\n\n\n")

#The type of inidicator where ME is MEteorology, QA Quality if Air, RU Noise and CT traffic control
print('ME - Metereologia | QA - Qualidade do Ar | RU - Ruído | CT - Contador de Tráfego')
print()

indicator = input("Indicador (XX): ") 
print("\n\n\n")

## The compounds and other specific data to retrieve
print("C6H6 - Benzeno | 00CO - Monóx Carbono | 0NO2 - Dióxido de Azoto | 00O3 - Ozono | PM10 | PM25 | 0SO2 - Dióxio de Enxofre")
print("TEMP - Temperatura | 0VTH - Volume Tráfego Horário | 00UV – Ultravioleta | 00VD – Direção do Vento | 00VI – Intensidade do Vento")
print()

parameter = input("Parâmetro: ")
print("\n\n")

url2 = "http://opendata-cml.qart.pt/measurements/" + indicator + parameter + local + "?startDate=" + date_start + "&endDate=" + date_end

print(url2)
print()
print("Aguarde sff...")

''' 
Sample URL
url2 = "https://opendata-cml.qart.pt/measurements/QAPM250023?startDate=202201010101&endDate=202202020202"
url2 = "https://opendata-cml.qart.pt/measurements/QA0NO20032?startDate=202101010101&endDate=202203290202"
url2 = "https://opendata-cml.qart.pt/measurements/QAPM250046?startDate=202103030101&endDate=202203030101"
'''


with urllib.request.urlopen(url2) as url:
    data = json.loads(url.read().decode())

length = len(data)
lenght_days = round(length / 24)

print()
print("Registos obtidos: " + str(length) + ", que correspondem a " + str(lenght_days) + " dias")
print("\n\n")

val = list()
zeros = 0
neg = 0
dqn = 0
val_neg = list()
dqn_list = list()

'''
Upper limit values accepet today
PM25 = 75
PM10 = 100
C6H6 = 75
00CO = 100
0NO2 = 400
00O3 = 1200
0SO2 = 1200
'''

if parameter == "PM25" or parameter == "C6H6":
    limit = 75

if parameter == "PM10" or parameter == "00CO":
    limit = 100

if parameter == "00O3" or parameter == "0SO2":
    limit = 1200

if parameter == "0NO2":
    limit = 400    
    
for item in data:
    
    if (item['value']) == 0:
        zeros += 1
    if (item['value']) < 0:
        neg += 1
        val_neg.append(item['value'])
    if (item['value']) < limit:
        val.append(item['value'])
    else:
        dqn += 1
        dqn_lista.append(item['value'])

#Print the number of values less than zero (errors) that will be eliminated
print("Valores inferiores a zero (eliminados): " + str(neg) + " (" + str(val_neg)[1:-1] + ")")
#print number of zeros
print("Sem registos ou zero: " + str(zeros))	
#print statistics
print("Média: " + str(statistics.mean(val)))
print("Mediana: " + str(statistics.median(val)))
print("Moda: " + str(statistics.mode(val)))
print("Máximo: " + str(max(val)))
print("Valores acima de " + str(limit) + ": " + str(dqn))


x = list()
i = 0

lists = [[] for _ in range(24)]

'''My old way:
l0,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23 = list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()'''


while i <= 23:
#    lista = str("l" + str(i))
#    print(lista)
#    lista = list()
    x.append(i)
    i += 1

## sum hourly
for item in data:
    data_time = item['date']
    hour = data_time[8:10]
 
hour_dict = {
    "00": l0,
    "01": l1,
    "02": l2,
    "03": l3,
    "04": l4,
    "05": l5,
    "06": l6,
    "07": l7,
    "08": l8,
    "09": l9,
    "10": l10,
    "11": l11,
    "12": l12,
    "13": l13,
    "14": l14,
    "15": l15,
    "16": l16,
    "17": l17,
    "18": l18,
    "19": l19,
    "20": l20,
    "21": l21,
    "22": l22,
    "23": l23
}

hour_dict[hour].append(item['value'])
 
'''
My old way also
    if hour == "01":
        l1.append(item['value'])
    if hour == "02":
        l2.append(item['value'])
    if hour == "03":
        l3.append(item['value'])
    if hour == "04":
        l4.append(item['value'])
    if hour == "05":
        l5.append(item['value'])
    if hour == "06":
        l6.append(item['value'])
    if hour == "07":
        l7.append(item['value'])
    if hour == "08":
        l8.append(item['value'])
    if hour == "09":
        l9.append(item['value'])
    if hour == "10":
        l10.append(item['value'])
    if hour == "11":
        l11.append(item['value'])
    if hour == "12":
        l12.append(item['value'])
    if hour == "13":
        l13.append(item['value'])
    if hour == "14":
        l14.append(item['value'])
    if hour == "15":
        l15.append(item['value'])
    if hour == "16":
        l16.append(item['value'])
    if hour == "17":
        l17.append(item['value'])
    if hour == "18":
        l18.append(item['value'])
    if hour == "19":
        l19.append(item['value'])
    if hour == "20":
        l20.append(item['value'])
    if hour == "21":
        l21.append(item['value'])
    if hour == "22":
        l22.append(item['value'])
    if hour == "23":
        l23.append(item['value'])
    if hour == "00":
        l0.append(item['value'])
'''

print("\n\n")

for hour, lst in enumerate([l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23]):
    print(f"Soma às {hour:02d}: {sum(lst)}")

'''
Old way    
print("Soma às " + "00: " + str(sum(l0)))
print("Soma às " + "01: " + str(sum(l1)))
print("Soma às " + "02: " + str(sum(l2)))
print("Soma às " + "03: " + str(sum(l3)))
print("Soma às " + "04: " + str(sum(l4)))
print("Soma às " + "05: " + str(sum(l5)))
print("Soma às " + "06: " + str(sum(l6)))
print("Soma às " + "07: " + str(sum(l7)))
print("Soma às " + "08: " + str(sum(l8)))
print("Soma às " + "09: " + str(sum(l9)))
print("Soma às " + "10: " + str(sum(l10)))
print("Soma às " + "11: " + str(sum(l11)))
print("Soma às " + "12: " + str(sum(l12)))
print("Soma às " + "13: " + str(sum(l13)))
print("Soma às " + "14: " + str(sum(l14)))
print("Soma às " + "15: " + str(sum(l15)))
print("Soma às " + "16: " + str(sum(l16)))
print("Soma às " + "17: " + str(sum(l17)))
print("Soma às " + "18: " + str(sum(l18)))
print("Soma às " + "19: " + str(sum(l19)))
print("Soma às " + "20: " + str(sum(l20)))
print("Soma às " + "21: " + str(sum(l21)))
print("Soma às " + "22: " + str(sum(l22)))
print("Soma às " + "23: " + str(sum(l23)))    
'''

y = [sum(lst) / len(lst) for lst in [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23]]

'''
Old way
y = [sum(l0)/len(l0), sum(l1)/len(l1),sum(l2)/len(l2),sum(l3)/len(l3),sum(l4)/len(l4),sum(l5)/len(l5),sum(l6)/len(l6),sum(l7)/len(l7),sum(l8)/len(l8),sum(l9)/len(l9),sum(l10)/len(l10),sum(l11)/len(l11),sum(l12)/len(l12), sum(l13)/len(l13), sum(l14)/len(l4), sum(l15)/len(l15), sum(l16)/len(l16),sum(l17)/len(l17),sum(l18)/len(l18),sum(l19)/len(l19),sum(l20)/len(l20),sum(l21)/len(l21),sum(l22)/len(l22), sum(l23)/len(l23)]
'''

print("\n\n")
input("Clicar [Enter] para ver gráfico da distribuição por 24h e depois FECHAR a janela")

#Plot the graph for 24h

plt.plot(x, y)
 
plt.xlabel('Horário')
plt.ylabel('Concentração Média')
plt.show()


#Plot the graph for the entire time
print("\n\n")
input("Clicar [Enter] para ver gráfico da distribuição por todo o tempo")

plt.clf()

xx = list()
ii = 0
for v in val:
    xx.append(ii)
    ii += 1
    

plt.plot(xx, val)

plt.xlabel('Dias')
plt.ylabel('Concentração')
plt.show()


print()
print()
export = input("Pretendes exportar para excel os dados gerais (S / N)?: ")

plt.clf()

xx.insert(0, "Dias")
val.insert(0, "Dados")

if export == "S" or export == "s" or export == "Sim" or export == "sim":
    df = pd.DataFrame({"Concentração": val, "Dia": xx})
    df.to_excel('dados.xlsx', index = False, sheet_name='Valores Globais')


#---------------------Tendencia - Reg Linear----------------------------------------

print()
print()

xx.pop(0)
val.pop(0)

x_x = np.array(xx) 
y_y = np.array(val)
n_n = np.size(x_x)
  
x_x_mean = np.mean(x_x)
y_y_mean = np.mean(y_y)
x_x_mean,y_y_mean
  
Sxy = np.sum(x_x*y_y)- n_n*x_x_mean*y_y_mean
Sxx = np.sum(x_x*x_x)-n_n*x_x_mean*x_x_mean
  
b1 = Sxy/Sxx
b0 = y_y_mean-b1*x_x_mean
print('Declive b1: ', b1)
print('Interceção b0: ', b0)
  
plt.scatter(x_x,y_y)
plt.xlabel('Dias')
plt.ylabel('Valores')


y_pred = b1 * x_x + b0
  
plt.scatter(x_x, y_y, color = 'red')
plt.plot(x_x, y_pred, color = 'green')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


error = y_y - y_pred
se = np.sum(error**2)
print('Squared error: ', se)
  
mse = se/n_n 
print('Mean squared error: ', mse)
  
rmse = np.sqrt(mse)
print('Root mean square error: ', rmse)
  
SSt = np.sum((y_y - y_y_mean)**2)
R2 = 1- (se/SSt)
print('R square is: ', R2)

x_x = x_x.reshape(-1,1)
regression_model = LinearRegression()
  
# Fit the data(train the model)
regression_model.fit(x_x, y_y)
  
# Predict
y_predicted = regression_model.predict(x_x)
  
# model evaluation
mse=mean_squared_error(y_y,y_predicted)
  
rmse = np.sqrt(mean_squared_error(y_y, y_predicted))
r2 = r2_score(y_y, y_predicted)
  
# printing values
print('Desclive: ' ,regression_model.coef_)
print('Interceção: ', regression_model.intercept_)
print('MSE: ',mse)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)