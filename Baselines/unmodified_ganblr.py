from ganblr import get_demo_data
from ganblr.models import GANBLR

# this is a discrete version of adult since GANBLR requires discrete data.
df = get_demo_data('adult')
x, y = df.values[:,:-1], df.values[:,-1]

model = GANBLR()
model.fit(x, y, epochs = 10)

#generate synthetic data
synthetic_data = model.sample(1000)

lr_result = model.evaluate(x, y, model='lr')
mlp_result = model.evaluate(x, y, model='mlp')
rf_result = model.evaluate(x, y, model='rf')

results = {
    "Logistic Regression": lr_result,
    "MLP": mlp_result,
    "Random Forest": rf_result
}

for model_name, result in results.items():
    print(f"{model_name}: {result}")