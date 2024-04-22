import requests
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


dataPath = r'../resourses/Data/'

def get_next_number(directory):
    files = os.listdir(directory)

    numbers = []

    for file in files:
        if file.endswith('.txt'):
            try:
                number = int(file.split('.')[0])
                numbers.append(number)
            except ValueError:
                pass

    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1

    return next_number

def get_movie_reviews(movie_id):
    api_url = f'https://kinopoiskapiunofficial.tech/api/v2.2/films/{movie_id}/reviews'
    headers = {'X-API-KEY': '7ae52701-d69b-4d57-9692-de04a3c81695'}
    response = requests.get(api_url, headers=headers, verify=False)
    if response.status_code == 200:
        reviews_data = response.json().get('items')
        return reviews_data
    else:
        print(f"Failed to fetch reviews for movie ID {movie_id}")
        return []

def save_review_to_file(review_text, review_type):
    directory = 'unknown'
    if review_type == "POSITIVE":
        directory = 'pos'
    elif review_type == "NEUTRAL":
        directory = 'neu'
    elif review_type == "NEGATIVE":
        directory = 'neg'

    directory = os.path.join(dataPath, directory)  # Обновляем путь к директории
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    n = get_next_number(directory)

    file_path = os.path.join(directory, f"{n}.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(review_text)


def main():
    for movie_id in range(1001, 2000):
        reviews = get_movie_reviews(movie_id)
        for review in reviews:
            review_text = review.get('description')
            r_type = review.get('type')
            if review_text:
                save_review_to_file(review_text, r_type)

if __name__ == "__main__":
    main()
