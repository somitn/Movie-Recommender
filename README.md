#Install all dependencies in VScode or any other
pip install pandas numpy scikit-learn
pip install scikit-surprise
pip install matplotlib seaborn

#Execute
python movie_recommender.py or just press the play button if on vscode

#Important Movies.csv and ratings.csv should be on the same directory of the movie_recommender.py

#movie.csv structure
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
Total movies: 9,742

Unique genres: Drama, Comedy, Action, Thriller, Romance, Horror, Sci-Fi, etc.

Movies can have multiple genres (pipe-separated)

#ratings.csv structure


Column Name	Data Type	Description	Example
userId	Integer	Unique identifier for each user (1-610)	1, 2, 3, 100, 610
movieId	Integer	Foreign key linking to movies.csv	1, 318, 260, 1196
rating	Float	User's rating from 0.5 to 5.0 (0.5 increments)	4.0, 5.0, 3.5, 0.5
timestamp	Integer	Unix timestamp when rating was given	964982703, 1445714835

Total ratings: 100,836

Number of users: 610

Number of movies: 9,742

Rating scale: 0.5 to 5.0 (in 0.5 increments)

Sparsity: 98.3% empty (most users haven't rated most movies)

