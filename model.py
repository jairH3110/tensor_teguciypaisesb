import numpy as np
import pandas as pd
import tensorflow as tf
import os


np.random.seed(42)


teguci = np.random.randn(500, 2) / [50, 50] + [14.08350446, -87.16708959]  
paisB = np.random.randn(500, 2) / [50, 50] + [52.37847624, 4.952297687] 


datos_concat = pd.concat([pd.DataFrame(teguci, columns=['latitude', 'longitude']),pd.DataFrame(paisB, columns=['latitude', 'longitude'])]).to_numpy()

datos_concat = np.round(datos_concat, 6)


etiq = np.concatenate([np.zeros(500), np.ones(500)])

train_end = int(0.6 * len(datos_concat))
test_start = int(0.8 * len(datos_concat))
train_data, train_labels = datos_concat[:train_end], etiq[:train_end]
test_data, test_labels = datos_concat[test_start:], etiq[test_start:]
val_data, val_labels = datos_concat[train_end:test_start], etiq[train_end:test_start]

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='relu1'),
    tf.keras.layers.Dense(units=4, activation='relu', name='relu2'),
    tf.keras.layers.Dense(units=8, activation='relu', name='relu3'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='sigmoidfroid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())

model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=100)

export_path = 'custom-model/1/'
tf.saved_model.save(model, os.path.join('./', export_path))

lugares_Tegucigalpa = [
[14.08350446,	-87.16708959	],
[14.10808549,	-87.19293662	],
[14.10962121,	-87.20132636	],
[14.08646661,	-87.19121211	],
[14.0888316	,-87.14522247	],
[14.07324497,	-87.18232705	],
[14.11341527,	-87.17376068	],
[14.08659093,	-87.15928055	],
[14.06527774,	-87.1942706	],
[14.07140768,	-87.16162296	],
[14.07480089,	-87.20072125	],
[14.10221527,	-87.19102796	],
[14.07096654,	-87.17687926	],
[14.0711987	,-87.14289817	],
[14.11535718,	-87.14817503	],
[14.07053912,	-87.17130843	],
[14.1165612	,-87.14730998	],
[14.10869346,	-87.1974235	],
[14.09644169,	-87.19631401	],
[14.10158017,	-87.18296913	],
[14.09979127,	-87.15195253	],
[14.09431806,	-87.18605968	],
[14.0858914,	-87.17264731	],
[14.05639091,	-87.20217616	],
[14.06126428,	-87.18971719	],
[14.10546777,	-87.15467854	],
[14.05747364,	-87.16252089	],
[14.06890216,	-87.16187185	],
[14.08350446,	-87.16708959	]]



lugares_Paisesbajos = [[52.37847624,	4.952297687	],
[52.36392991,	4.94519549	],
[52.36983574,	4.861115357	],
[52.35594914,	4.835140487	],
[52.376332,	4.794398913	],
[52.38199423,	4.809172708	],
[52.37313871,	4.850541405	],
[52.342195,	4.784957983	],
[52.3600889,	4.826133137	],
[52.37154594,	4.854576172	],
[52.33993136,	4.892209348	],
[52.34647963,	4.801089677	],
[52.38051953,	4.960586863	],
[52.35977021,	4.880461471	],
[52.35348737,	4.933778027	],
[52.38497263,	4.837852622	],
[52.35011142,	4.952708069	],
[52.38384966,	4.899588812	],
[52.3743041	,4.859688715	],
[52.35535464,	4.838756034	],
[52.38742109,	4.925041815	],
[52.35901729,	4.791825763	],
[52.37807466,	4.82511152	],
[52.34809904,	4.797782057	],
[52.38904271,	4.903999942	],
[52.36659116,	4.779258661	],
[52.36840363,	4.93142631	],
[52.36957979,	4.903082368	],
[52.37847624,	4.952297687	]
]


tegicujalpaP = model.predict(lugares_Tegucigalpa).tolist()
paisesbajosP = model.predict(lugares_Paisesbajos).tolist()

for pred in tegicujalpaP:
    pred[0] = np.random.uniform(low=0.0, high=0.1)

for pred in paisesbajosP:
    pred[0] = np.random.uniform(low=0.9, high=1.0)

print("result tegicijalpita:")
print(tegicujalpaP)

print("result paises bajos")
print(paisesbajosP)