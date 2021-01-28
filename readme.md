Env Variables:

```
SAVE_MASK
```
Set to true of false if you want to save mask during training session


PLAN FOR LAYERS

take layer types as param to builder [done]
pass types to neunet constructor [done]
neunet constructor iterates through and creates each layer and their type
implement interface for layers


overall schema
private __eval methods call layer specific act functions
class layernet needs to have each layer object saved
cost function calls specific layer cost