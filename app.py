from flask import Flask, render_template, request

app = Flask(__name__)

# Example function to fetch news articles
def fetch_news_articles():
    # Replace this with your actual logic to fetch news articles
    return [
        {
            'title': 'Health Benefits of Regular Exercise',
            'description': 'Regular exercise can improve your health in many ways.',
            'image': 'path/to/image.jpg',
            'publishedAt': '2023-10-01',
            'source': {'name': 'Health News'},
            'url': 'https://example.com/article1'
        },
        # Add more articles as needed
    ]

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(f"Received symptoms: {symptoms}")
        return "Form submitted!"
    return render_template("index.html")

@app.route('/news')
def news():
    news_articles = fetch_news_articles()  # Fetch news articles
    return render_template('news.html', news_articles=news_articles)

if __name__ == "__main__":
    app.run(debug=True) 