from flask import Flask, render_template, request
from deeplearning import deeplearning

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def update():
	if request.method == 'POST':
		foodTitles = []
		# PUT IN FAULT TOLERANCE
		calories = request.form['calories']
		try:
			calories = int(calories)
		except ValueError as ex:
			message = str(ex) + ": requires an integer"
			return render_template('home.html')

		overUnder = request.form.get('overUnder')

		restrictions = request.form.get('restrictions')

		allergies = []
		list = ['dairy','eggs','tree','peanuts','shell','soy','fish','gluten']
		for allergy in list:
			if request.form.get(allergy) is not None:
				allergies.append(request.form.get(allergy))

		foodTitles, foodRatings = deeplearning(calories, overUnder, restrictions, allergies)
		foodTitles = foodTitles.values.tolist()
		foodRatings = foodRatings.values.tolist()

		for i in range(len(foodTitles)):
			try:
				foodTitles[i] = foodTitles[i].encode('ascii', 'ignore').decode('ascii')
				foodRatings[i] = "%.4f" % foodRatings[i]
				foodTitles[i] = foodTitles[i] + ": " + str(foodRatings[i]) + "%"
			except AttributeError as ex:
				return render_template('home.html')


		return render_template('home.html', food = foodTitles)

	return render_template('home.html')
