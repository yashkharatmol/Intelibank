from app.pipeline import combined_pipeline

if __name__ == "__main__":
    text = input("Enter your query: ")
    print(combined_pipeline(text))
