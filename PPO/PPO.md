```mermaid
flowchart TD
    Start([PPO Algorithm - What is 2+2?])
    
    Start --> Setup[SETUP<br/>gamma = 0.99<br/>lambda = 0.95<br/>epsilon = 0.2]
    
    Setup --> GenTitle[STEP 1: SEQUENTIAL GENERATION<br/>Generate token by token]
    
    GenTitle --> S0[Step t=0: Generate T<br/>State: What is 2+2?<br/>Action: T<br/>log_prob = -2.5<br/>reward = 0]
    
    GenTitle --> S1[Step t=1: Generate h<br/>State: What is 2+2? T<br/>Action: h<br/>log_prob = -2.1<br/>reward = 0]
    
    GenTitle --> S2[Step t=2: Generate e<br/>State: What is 2+2? Th<br/>Action: e<br/>log_prob = -1.8<br/>reward = 0]
    
    GenTitle --> S3[Step t=3: Generate space<br/>State: What is 2+2? The<br/>Action: space<br/>log_prob = -1.5<br/>reward = 0]
    
    GenTitle --> S4[Step t=4: Complete<br/>State: What is 2+2? The<br/>Action: answer<br/>log_prob = -1.2<br/>reward = +10]
    
    S0 --> CriticTitle[STEP 2: CRITIC PREDICTS VALUES<br/>Estimate future rewards at each state]
    S1 --> CriticTitle
    S2 --> CriticTitle
    S3 --> CriticTitle
    S4 --> CriticTitle
    
    CriticTitle --> V0[Value at t=0:<br/>V s0 = 4.0<br/>Expected future reward]
    
    CriticTitle --> V1[Value at t=1:<br/>V s1 = 5.0<br/>Expected future reward]
    
    CriticTitle --> V2[Value at t=2:<br/>V s2 = 6.0<br/>Expected future reward]
    
    CriticTitle --> V3[Value at t=3:<br/>V s3 = 7.5<br/>Expected future reward]
    
    CriticTitle --> V4[Value at t=4:<br/>V s4 = 9.0<br/>Expected future reward]
    
    CriticTitle --> V5[Value at t=5:<br/>V s5 = 0<br/>Terminal state]
    
    V0 --> TDTitle[STEP 3: COMPUTE TD ERRORS<br/>Forward calculation using next state value]
    V1 --> TDTitle
    V2 --> TDTitle
    V3 --> TDTitle
    V4 --> TDTitle
    V5 --> TDTitle
    
    TDTitle --> TD0[TD Error at t=0:<br/>delta0 = r0 + gamma x V s1 - V s0<br/>delta0 = 0 + 0.99 x 5.0 - 4.0<br/>delta0 = 4.95 - 4.0<br/>delta0 = 0.95]
    
    TDTitle --> TD1[TD Error at t=1:<br/>delta1 = r1 + gamma x V s2 - V s1<br/>delta1 = 0 + 0.99 x 6.0 - 5.0<br/>delta1 = 5.94 - 5.0<br/>delta1 = 0.94]
    
    TDTitle --> TD2[TD Error at t=2:<br/>delta2 = r2 + gamma x V s3 - V s2<br/>delta2 = 0 + 0.99 x 7.5 - 6.0<br/>delta2 = 7.425 - 6.0<br/>delta2 = 1.425]
    
    TDTitle --> TD3[TD Error at t=3:<br/>delta3 = r3 + gamma x V s4 - V s3<br/>delta3 = 0 + 0.99 x 9.0 - 7.5<br/>delta3 = 8.91 - 7.5<br/>delta3 = 1.41]
    
    TDTitle --> TD4[TD Error at t=4:<br/>delta4 = r4 + gamma x V s5 - V s4<br/>delta4 = 10 + 0.99 x 0 - 9.0<br/>delta4 = 10 - 9.0<br/>delta4 = 1.0]
    
    TD0 --> GAETitle[STEP 4: COMPUTE GAE ADVANTAGES<br/>BACKWARD from last step!<br/>Uses future advantages]
    TD1 --> GAETitle
    TD2 --> GAETitle
    TD3 --> GAETitle
    TD4 --> GAETitle
    
    GAETitle --> A4[Advantage at t=4 LAST:<br/>A4 = delta4<br/>A4 = 1.0<br/>No future to consider]
    
    GAETitle --> A3[Advantage at t=3:<br/>A3 = delta3 + gamma x lambda x A4<br/>A3 = 1.41 + 0.99 x 0.95 x 1.0<br/>A3 = 1.41 + 0.9405<br/>A3 = 2.351<br/>Uses A4 from future!]
    
    GAETitle --> A2[Advantage at t=2:<br/>A2 = delta2 + gamma x lambda x A3<br/>A2 = 1.425 + 0.99 x 0.95 x 2.351<br/>A2 = 1.425 + 2.210<br/>A2 = 3.635<br/>Uses A3 which contains A4!]
    
    GAETitle --> A1[Advantage at t=1:<br/>A1 = delta1 + gamma x lambda x A2<br/>A1 = 0.94 + 0.99 x 0.95 x 3.635<br/>A1 = 0.94 + 3.419<br/>A1 = 4.359<br/>Uses A2 which contains A3 A4!]
    
    GAETitle --> A0[Advantage at t=0 FIRST:<br/>A0 = delta0 + gamma x lambda x A1<br/>A0 = 0.95 + 0.99 x 0.95 x 4.359<br/>A0 = 0.95 + 4.099<br/>A0 = 5.049<br/>Uses A1 which contains ALL future!]
    
    A0 --> LossTitle[STEP 5: POLICY LOSSES<br/>Compute loss for each timestep]
    A1 --> LossTitle
    A2 --> LossTitle
    A3 --> LossTitle
    A4 --> LossTitle
    
    LossTitle --> L0[Loss at t=0 for T with A0=5.049:<br/>New log_prob = -2.3<br/>Old log_prob = -2.5<br/>ratio = exp-2.3 / exp-2.5 = 1.22<br/>CLIPPED to 1.2 epsilon limit<br/>surr1 = 1.22 x 5.049 = 6.160<br/>surr2 = 1.2 x 5.049 = 6.059<br/>L0 = -min 6.160, 6.059 = -6.059<br/>Increase probability!]
    
    LossTitle --> L1[Loss at t=1 for h with A1=4.359:<br/>ratio = 1.15<br/>surr1 = 1.15 x 4.359 = 5.013<br/>surr2 = 1.2 x 4.359 = 5.231<br/>L1 = -5.013<br/>Increase probability!]
    
    LossTitle --> L2[Loss at t=2 for e with A2=3.635:<br/>ratio = 1.10<br/>L2 = -4.000<br/>Increase probability!]
    
    LossTitle --> L3[Loss at t=3 for space with A3=2.351:<br/>ratio = 1.08<br/>L3 = -2.539<br/>Increase probability!]
    
    LossTitle --> L4[Loss at t=4 for answer with A4=1.0:<br/>ratio = 1.05<br/>L4 = -1.050<br/>Increase probability!]
    
    L0 --> VLossTitle[STEP 6: VALUE LOSSES<br/>Train critic to predict better]
    L1 --> VLossTitle
    L2 --> VLossTitle
    L3 --> VLossTitle
    L4 --> VLossTitle
    
    VLossTitle --> VL[Value Loss Examples:<br/>Target at t=0 = A0 + V s0 = 5.049 + 4.0 = 9.049<br/>Predicted = 4.0<br/>VL0 = 4.0 - 9.049 squared = 25.49<br/>Similar for other timesteps<br/>Critic learns to predict better!]
    
    VL --> Update[STEP 7: UPDATE NETWORKS<br/>Total Loss = Policy Loss + 0.5 x Value Loss<br/>loss backward<br/>optimizer step<br/>Update Actor weights<br/>Update Critic weights]
    
    Update --> Summary[GAE RESULT:<br/>A0 = 5.049 has all future steps<br/>A1 = 4.359 has t=1,2,3,4<br/>A2 = 3.635 has t=2,3,4<br/>A3 = 2.351 has t=3,4<br/>A4 = 1.0 only t=4<br/>Early steps get credit for final reward!<br/>Backward propagation through advantages]
    
    Summary --> End([PPO COMPLETE<br/>5 sequential steps<br/>Critic predicted values<br/>GAE backward computation<br/>Updated 2 networks Actor and Critic])
    
    style Start fill:#ff9999,stroke:#ff0000,stroke-width:4px
    style S0 fill:#ffe6e6
    style S1 fill:#ffe6e6
    style S2 fill:#ffe6e6
    style S3 fill:#ffe6e6
    style S4 fill:#ffe6e6
    style CriticTitle fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style V0 fill:#ffd9d9
    style V1 fill:#ffd9d9
    style V2 fill:#ffd9d9
    style V3 fill:#ffd9d9
    style V4 fill:#ffd9d9
    style V5 fill:#ffcccc
    style TDTitle fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style TD0 fill:#ffe6e6
    style TD1 fill:#ffe6e6
    style TD2 fill:#ffe6e6
    style TD3 fill:#ffe6e6
    style TD4 fill:#ffe6e6
    style GAETitle fill:#ff9999,stroke:#ff0000,stroke-width:3px
    style A4 fill:#ffeeee
    style A3 fill:#ffd9d9
    style A2 fill:#ffc4c4
    style A1 fill:#ffafaf
    style A0 fill:#ff9999,stroke:#ff0000,stroke-width:3px
    style LossTitle fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style L0 fill:#ffe6e6
    style L1 fill:#ffe6e6
    style L2 fill:#ffe6e6
    style L3 fill:#ffe6e6
    style L4 fill:#ffe6e6
    style VLossTitle fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style VL fill:#ffd9d9
    style Update fill:#ff9999,stroke:#ff0000,stroke-width:3px
    style Summary fill:#fff0f0,stroke:#ff6600,stroke-width:3px
    style End fill:#ffeeee,stroke:#ff0000,stroke-width:4px