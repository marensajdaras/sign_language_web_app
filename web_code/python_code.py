from flask import Flask, render_template, request, redirect, url_for
from flask_dropzone import Dropzone
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'  # Use your preferred database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
dropzone = Dropzone(app)


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)

db.create_all()


@app.route('/')
def index():
    videos = Video.query.all()
    return render_template('index.html', videos=videos)


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if f:
        filename = f.filename
        video = Video(filename=filename)
        db.session.add(video)
        db.session.commit()
        f.save('uploads/' + filename)  # Save the file to a folder (create the folder if not exists)

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
