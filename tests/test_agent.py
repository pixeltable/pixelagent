from pixelagent import Agent

def test_basic_response():
    agent = Agent()
    response = agent.run("Hello!")
    print(response)
    assert isinstance(response, str)
    assert len(response) > 0