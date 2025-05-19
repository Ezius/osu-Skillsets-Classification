# osu-Skillsets-Classification
My personal Project to train an AI to classify the skillsets of an osu beatmap

Ideas & Todos
``` mermaid
kanban
    Mods and Skillsets
        Mod Flags to extra_data
        New Function to change Data based on mods for DT, HT, HR, EZ
        Think of ways to represent extra Skillsets and how to better represent Reading
        Multiskillset Labeling

    Model
        Improve current primituve Model architecture
        Klassification with a Transformer
        Variational Autoencoder to experiment with new Concepts of Skillsets

    Refactoring
        Seperate File for all the Webrequest stuff
        Code Cleanup and Optimization
        Code Optimization for Circle function or other stuff in beatmap_parser
        query_beatmap contains some duplicates again add shared seen_ids list across Processes
        query_beatmap also has an unequal amount of work for every core but not a huge priority
        
```