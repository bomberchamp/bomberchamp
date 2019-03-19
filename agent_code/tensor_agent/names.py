from random import choice

left = ["adoring", "bold", "clever", "confident", "crazy", "dreamy", "flamboyant", "hardcore", "optimistic", "stoic", "vigilant", "zealous"]

center = ["bomb", "coin", "crate"]

right = ["advocate", "collector", "lover", "researcher"]

used_names = {}

def get_random_name():
    name = f'{choice(left)}_{choice(center)}_{choice(right)}'
    if name in used_names:
        used_names[name] += 1
        return f'{name}{used_names[name]}'
    else:
        used_names[name] = 0
        return name
