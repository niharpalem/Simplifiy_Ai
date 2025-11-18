```mermaid
flowchart TD
  Start([DPO Training Example])
  Start --> Input[Input: What is 2+2?<br/>Chosen answer yw: 4<br/>Rejected answer yl: 3]
  
  Input --> Step1[Step 1: Evaluate both models]
  Step1 --> Ref[Reference Model πref FIXED<br/>log πref of yw = -1.609<br/>log πref of yl = -2.303]
  Step1 --> Pol[Policy Model πθ TRAINABLE<br/>log πθ of yw = -2.996<br/>log πθ of yl = -1.897]
  
  Ref --> Step2[Step 2: Compute advantages<br/>A of y = log πθ of y − log πref of y]
  Pol --> Step2
  
  Step2 --> Adv[A of yw = -2.996 − -1.609 = -1.387<br/>A of yl = -1.897 − -2.303 = +0.406]
  
  Adv --> Step3[Step 3: Compute pair difference<br/>Δ = β · A of yw − A of yl<br/>with β = 1.0]
  
  Step3 --> Delta[Δ = -1.387 − 0.406 = -1.793]
  
  Delta --> Step4[Step 4: Compute loss<br/>L = −log of 1 / 1 + exp of −Δ]
  
  Step4 --> Loss[L = −log of 1 / 1 + exp of 1.793<br/>L = −log of 0.143<br/>L = 1.946]
  
  Loss --> Step5[Step 5: Update πθ with gradient descent<br/>πref stays fixed]
  
  Step5 --> After[After Update:<br/>log πθ of yw = -1.715 improved<br/>log πθ of yl = -2.526 worsened]
  
  After --> Verify[Verify: Recompute loss]
  Verify --> NewAdv[New A of yw = -0.106<br/>New A of yl = -0.223<br/>New Δ = +0.118]
  
  NewAdv --> NewLoss[New L = 0.635]
  
  NewLoss --> Result([Success!<br/>Loss: 1.946 → 0.635<br/>Model prefers correct answer])
  
  style Start fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
  style Input fill:#fff3e0
  style Step1 fill:#f3e5f5
  style Ref fill:#e8f5e9
  style Pol fill:#ffebee
  style Step2 fill:#fff9c4
  style Step3 fill:#ffe0b2
  style Step4 fill:#ffccbc
  style Step5 fill:#ce93d8
  style After fill:#c8e6c9
  style Verify fill:#b2dfdb
  style Result fill:#a5d6a7,stroke:#2e7d32,stroke-width:3px