from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from pydantic import BaseModel, Field

# 1. Define your desired response structure using Pydantic
class MovieMetadata(BaseModel):
    title: str = Field(description="The formal title of the movie.")
    release_year: int = Field(description="The year the movie was released.")
    genres: list[str] = Field(description="List of genres associated with the movie.")
    director: str = Field(description="The full name of the director.")
    written_by: str = Field(description="The full name of the script writer.")
    actors: list[str] = Field(description="List of main actors name in the movie.")
    box_office_usd: float = Field(description="Total lifetime box office gross revenue in USD.")

# 2. Initialize the Anthropic client
client = Anthropic()

# 3. Request structured data using 'output_config'
response = client.messages.parse(
    model="claude-haiku-5-5",
    max_tokens=500,
    messages=[
        {
            "role": "user",
            "content": "Extract data for the movie Cast Away directed by Robert Zemeckis."
        }
    ],
    # Enforce structured output formatting
    output_format=MovieMetadata
)

# 4. Access your validated data natively
movie_data = response.parsed_output

print(f"Title: {movie_data.title}")
print(f"Director: {movie_data.director}")
print(f"Year: {movie_data.release_year}")

# Print the validated dictionary output
clean_json_output = movie_data.model_dump_json(indent=2)
print("\nJSON Output: ", clean_json_output)
