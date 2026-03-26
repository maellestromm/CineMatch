from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QMainWindow, QPushButton, QStackedWidget, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CineMatch")
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        #home page
        mainPage = QWidget()
        layoutMain = QVBoxLayout(mainPage)
        self.central_widget.setLayout(layoutMain)
        title = QLabel("CineMatch")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 48px; font-weight: bold;")
        layoutMain.addWidget(title)
        buttonHolder = QWidget()
        buttonLayout = QHBoxLayout(buttonHolder)
        recommendByProfileButton = QPushButton("Recommend by Profile")
        buttonLayout.addWidget(recommendByProfileButton)
        recommendByProfileButton.setStyleSheet("font-size: 24px;")
        recommendByProfileButton.clicked.connect(lambda: self.central_widget.setCurrentIndex(1))
        recommendByMovieButton = QPushButton("Recommend by Movie")
        buttonLayout.addWidget(recommendByMovieButton)
        recommendByMovieButton.setStyleSheet("font-size: 24px;")
        recommendByMovieButton.clicked.connect(lambda: self.central_widget.setCurrentIndex(2))
        layoutMain.addWidget(buttonHolder)

        #page for recommending by profile
        byProfilePage = QWidget()
        layoutByProfile = QVBoxLayout(byProfilePage)
        profileInstruction = QLabel("Enter Letterboxd username: ")
        profileInstruction.setStyleSheet("font-size: 24px;")
        layoutByProfile.addWidget(profileInstruction)
        profileInput = QLineEdit()
        profileInput.setStyleSheet("font-size: 24px;")
        layoutByProfile.addWidget(profileInput)
        byProfileButtonHolder = QWidget()
        byProfileButtonLayout = QHBoxLayout(byProfileButtonHolder)
        backByProfileButton = QPushButton("Back")
        backByProfileButton.setStyleSheet("font-size: 24px;")
        byProfileButtonLayout.addWidget(backByProfileButton)
        backByProfileButton.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        submitProfileButton = QPushButton("Submit")
        submitProfileButton.setStyleSheet("font-size: 24px;")
        byProfileButtonLayout.addWidget(submitProfileButton)
        submitProfileButton.clicked.connect(lambda: self.showRecommendationsByProfile(profileInput))
        layoutByProfile.addWidget(byProfileButtonHolder)

        #page for recommending by movie
        byMoviePage = QWidget()
        layoutByMovie = QVBoxLayout(byMoviePage)
        movieInstruction = QLabel("Enter movie title: ")
        movieInstruction.setStyleSheet("font-size: 24px;")
        layoutByMovie.addWidget(movieInstruction)
        movieInput = QLineEdit()
        movieInput.setStyleSheet("font-size: 24px;")
        layoutByMovie.addWidget(movieInput)
        byMovieButtonHolder = QWidget()
        byMovieButtonLayout = QHBoxLayout(byMovieButtonHolder)
        backByMovieButton = QPushButton("Back")
        backByMovieButton.setStyleSheet("font-size: 24px;")
        byMovieButtonLayout.addWidget(backByMovieButton)
        backByMovieButton.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        submitMovieButton = QPushButton("Submit")
        submitMovieButton.setStyleSheet("font-size: 24px;")
        byMovieButtonLayout.addWidget(submitMovieButton)
        submitMovieButton.clicked.connect(lambda: self.showRecommendationsByMovie(movieInput))
        layoutByMovie.addWidget(byMovieButtonHolder)

        #loading screen
        loadingPage = QWidget()
        loadingLayout = QVBoxLayout(loadingPage)
        loadingLabel = QLabel("Loading...")
        loadingLabel.setAlignment(Qt.AlignCenter)
        loadingLabel.setStyleSheet("font-size: 24px; font-weight: bold;")
        loadingLayout.addWidget(loadingLabel)

        self.central_widget.addWidget(mainPage)
        self.central_widget.addWidget(byProfilePage)
        self.central_widget.addWidget(byMoviePage)
        self.central_widget.addWidget(loadingPage)

    def showRecommendationsByProfile(self, entryField):
        # The idea is, the function takes the username and runs the crawler to scrape data from their profile (including only movies in the database)
        # Then it runs the models to generate recommendations, and then displays those recommendations.
        self.central_widget.setCurrentIndex(3) # switch to loading screen temporarily
        profileName = entryField.text()
        pass

    def showRecommendationsByMovie(self, entryField):
        # ok this one should check if the movie is in the database first. then if it is, generate recommendations and display them
        self.central_widget.setCurrentIndex(3)
        movieName = entryField.text()
        pass

app = QApplication([])
window = MainWindow()
window.show()
app.exec()