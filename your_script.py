from openai import OpenAI

client = OpenAI(
    base_url="https://paid-ddc.xiolabs.xyz/v1",
    api_key="ddc.xiolabs.xyz/v1/status/provider-2"
)

def check_api_key():
    try:
        models = client.models.list()
        print("\nModels:")
        for model in models:
            print(model)
    except Exception as e:
        print("Error:", e)

# Call the function to check if the API key is working
check_api_key()