```mermaid
flowchart TD
    Start([GRPO Algorithm - What is 2+2?])
    
    Start --> Setup[SETUP<br/>Group Size K = 4<br/>epsilon = 0.2]
    
    Setup --> GenTitle[STEP 1: GENERATE GROUP<br/>4 complete responses at once]
    
    GenTitle --> R1[Response 1: 4<br/>Complete answer!<br/>log_prob = -1.2<br/>No sequential steps]
    
    GenTitle --> R2[Response 2: 5<br/>Complete answer!<br/>log_prob = -1.8<br/>Independent response]
    
    GenTitle --> R3[Response 3: four<br/>Complete answer!<br/>log_prob = -2.1<br/>Text form]
    
    GenTitle --> R4[Response 4: 3<br/>Complete answer!<br/>log_prob = -1.9<br/>Wrong answer]
    
    R1 --> ScoreTitle[STEP 2: REWARD MODEL SCORING<br/>Score each complete response]
    R2 --> ScoreTitle
    R3 --> ScoreTitle
    R4 --> ScoreTitle
    
    ScoreTitle --> Score1[Score 1:<br/>Input: What is 2+2?, 4<br/>R1 = +10.0<br/>Correct!]
    
    ScoreTitle --> Score2[Score 2:<br/>Input: What is 2+2?, 5<br/>R2 = -2.0<br/>Wrong]
    
    ScoreTitle --> Score3[Score 3:<br/>Input: What is 2+2?, four<br/>R3 = +3.0<br/>Acceptable]
    
    ScoreTitle --> Score4[Score 4:<br/>Input: What is 2+2?, 3<br/>R4 = -2.0<br/>Wrong]
    
    Score1 --> Mean[STEP 3: COMPUTE GROUP MEAN<br/>Rewards = 10, -2, 3, -2<br/>Baseline = sum / 4<br/>Baseline = 9 / 4<br/>Baseline = 2.25<br/>Simple arithmetic mean!<br/>No critic needed!]
    Score2 --> Mean
    Score3 --> Mean
    Score4 --> Mean
    
    Mean --> AdvTitle[STEP 4: COMPUTE ADVANTAGES<br/>Simple subtraction from baseline]
    
    AdvTitle --> Adv1[Advantage 1:<br/>A1 = R1 - baseline<br/>A1 = 10 - 2.25<br/>A1 = +7.75<br/>Much better than average!<br/>Strongly reinforce]
    
    AdvTitle --> Adv2[Advantage 2:<br/>A2 = R2 - baseline<br/>A2 = -2 - 2.25<br/>A2 = -4.25<br/>Much worse than average!<br/>Strongly suppress]
    
    AdvTitle --> Adv3[Advantage 3:<br/>A3 = R3 - baseline<br/>A3 = 3 - 2.25<br/>A3 = +0.75<br/>Slightly better<br/>Weak reinforcement]
    
    AdvTitle --> Adv4[Advantage 4:<br/>A4 = R4 - baseline<br/>A4 = -2 - 2.25<br/>A4 = -4.25<br/>Much worse than average!<br/>Strongly suppress]
    
    Adv1 --> LossTitle[STEP 5: POLICY LOSSES<br/>Compute loss for each response]
    Adv2 --> LossTitle
    Adv3 --> LossTitle
    Adv4 --> LossTitle
    
    LossTitle --> Loss1[Loss 1: For 4 with A1=+7.75<br/>ratio = 1.105<br/>surr1 = 1.105 x 7.75 = 8.564<br/>surr2 = 1.2 x 7.75 = 9.3<br/>L1 = -8.564<br/>Increase probability!]
    
    LossTitle --> Loss2[Loss 2: For 5 with A2=-4.25<br/>ratio = 1.105<br/>surr1 = 1.105 x -4.25 = -4.70<br/>surr2 = 1.2 x -4.25 = -5.1<br/>L2 = +4.70<br/>Decrease probability!]
    
    LossTitle --> Loss3[Loss 3: For four with A3=+0.75<br/>L3 = -0.8<br/>Slight increase]
    
    LossTitle --> Loss4[Loss 4: For 3 with A4=-4.25<br/>L4 = +4.70<br/>Decrease probability!]
    
    Loss1 --> Update[STEP 6: UPDATE NETWORK<br/>loss backward<br/>optimizer step<br/>Update Actor ONLY<br/>NO Critic!<br/>NO Value Loss!]
    Loss2 --> Update
    Loss3 --> Update
    Loss4 --> Update
    
    Update --> Summary[ADVANTAGE RESULT:<br/>A1 = +7.75 best response<br/>A2 = -4.25 worst response<br/>A3 = +0.75 acceptable<br/>A4 = -4.25 worst response<br/>Compared to group average!<br/>No future steps needed!]
    
    Summary --> End([GRPO COMPLETE<br/>4 complete responses<br/>Simple group mean<br/>No future steps<br/>Updated 1 network])
    
    style Start fill:#99ff99,stroke:#00cc00,stroke-width:4px
    style R1 fill:#e6ffe6
    style R2 fill:#e6ffe6
    style R3 fill:#e6ffe6
    style R4 fill:#e6ffe6
    style ScoreTitle fill:#ccffcc
    style Score1 fill:#d9f7d9
    style Score2 fill:#ffd9d9
    style Score3 fill:#d9f7d9
    style Score4 fill:#ffd9d9
    style Mean fill:#99ff99,stroke:#00cc00,stroke-width:3px
    style AdvTitle fill:#ccffcc
    style Adv1 fill:#b3ffb3
    style Adv2 fill:#ffb3b3
    style Adv3 fill:#d9f7d9
    style Adv4 fill:#ffb3b3
    style LossTitle fill:#ccffcc
    style Loss1 fill:#e6ffe6
    style Loss2 fill:#ffe6e6
    style Loss3 fill:#e6ffe6
    style Loss4 fill:#ffe6e6
    style Update fill:#99ff99,stroke:#00cc00,stroke-width:3px
    style Summary fill:#f0fff0,stroke:#00cc00,stroke-width:3px
    style End fill:#eeffee,stroke:#00cc00,stroke-width:4px