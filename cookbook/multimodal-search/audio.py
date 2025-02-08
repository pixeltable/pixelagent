import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

from pxl.agent import openai_agent


DIRECTORY = 'audio_index'
TABLE_NAME = f'{DIRECTORY}.audio'
VIEW_NAME = f'{DIRECTORY}.audio_sentence_chunks'
DELETE_INDEX = False

if DELETE_INDEX:
    pxt.drop_table(TABLE_NAME, force=True)

if TABLE_NAME not in pxt.list_tables():
    # Create audio table
    pxt.create_dir(DIRECTORY)
    audio_index = pxt.create_table(TABLE_NAME, {'audio_file': pxt.Audio})

    # Create audio-to-text column
    audio_index.add_computed_column(transcription=whisper.transcribe(audio=audio_index.audio_file, model='base.en'))

    # Create view that chunks text into sentences
    sentences_view = pxt.create_view(
        VIEW_NAME,
        audio_index,
        iterator=StringSplitter.create(text=audio_index.transcription.text, separators='sentence'),
    )

    # Define the embedding model
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

    # Create embedding index
    sentences_view.add_embedding_index(column='text', string_embed=embed_model)

else:
    audio_index = pxt.get_table(TABLE_NAME)


# Add data to the table
audio_index.insert([{'audio_file': 's3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3'}])

# Create a new agent

# Initialize the dog trainer agent
openai_agent.init(
    agent_name="Dog_Trainer",
    system_prompt="You specialize in training dogs",
    model_name="gpt-4o-mini",
    reset_memory=False,
)

# Run the agent
result = openai_agent.run(
    agent_name="Dog_Trainer", message="in 5 words tell me how to train my dog to sit"
)
print(result)