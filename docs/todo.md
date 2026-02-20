# TODO

## Current Working List

- Move the audio to a seperate thread, so that input can continue or queue.
- Consider how to interrupt ada to stop speaking. Perhaps a queue or bus we send a stop command to?
- Does voice synthesis take so much gpu resource, that we can't do inference. Does the next prompt in queue need to wait for speach to complete?
- Command to repeat the last thing Ada says, /repeat
- Audit all the agent say calls, to make sure Ada isn't 'talking' when she doesn't need to e.g. help output, menus, code blocks
- Response objects should have a .body method for all text, then a speakable method for text that should be spoken, again to not read code blocks
- Think about how to get better phonetic answers, for instance can we return phonemes alongside text in the llm output, could that simply be infered by an llm pass after?
