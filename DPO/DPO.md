```mermaid
flowchart TD
  Start([DPO Training Example])
  Start --> Input[Input: What is 2+2?<br/>Chosen answer yw: 4<br/>Rejected answer yl: 3<br/>Temperature β: 1.0]
  
  Input --> Formula[DPO Loss Formula:<br/>L = −log σ of β·log πθ of yw / πref of yw − β·log πθ of yl / πref of yl]
  
  Formula --> Step1[Step 1: Get log probabilities]
  
  Step1 --> Ref[Reference Model πref FIXED<br/>log πref of yw given x = -1.609<br/>log πref of yl given x = -2.303]
  
  Step1 --> Pol[Policy Model πθ TRAINABLE<br/>log πθ of yw given x = -2.996<br/>log πθ of yl given x = -1.897]
  
  Ref --> Step2[Step 2: Compute log ratios<br/>log ratio = log πθ − log πref]
  Pol --> Step2
  
  Step2 --> Ratios[For chosen yw:<br/>β·log πθ of yw / πref of yw = 1.0 · -2.996 − -1.609 = -1.387<br/><br/>For rejected yl:<br/>β·log πθ of yl / πref of yl = 1.0 · -1.897 − -2.303 = +0.406]
  
  Ratios --> Step3[Step 3: Compute difference Δ<br/>Δ = log ratio of yw − log ratio of yl]
  
  Step3 --> Delta[Δ = -1.387 − 0.406 = -1.793<br/><br/>Negative Δ means policy prefers WRONG answer!]
  
  Delta --> Step4[Step 4: Apply sigmoid<br/>σ of Δ = 1 / 1 + exp of −Δ]
  
  Step4 --> Sigmoid[σ of -1.793 = 1 / 1 + exp of 1.793<br/>= 1 / 1 + 6.00<br/>≈ 0.143<br/><br/>Only 14.3% confidence in chosen answer]
  
  Sigmoid --> Step5[Step 5: Compute loss<br/>L = −log σ of Δ]
  
  Step5 --> Loss[L = −log of 0.143 = 1.946<br/><br/>High loss = bad predictions]
  
  Loss --> Step6[Step 6: Backprop gradient ∂L/∂θ<br/>Gradient increases log πθ of yw<br/>Gradient decreases log πθ of yl]
  
  Step6 --> Update[Step 7: Gradient descent update<br/>θ ← θ − learning_rate · ∂L/∂θ<br/>πref stays FIXED]
  
  Update --> After[After Update:<br/>log πθ of yw = -1.715 improved!<br/>log πθ of yl = -2.526 worsened!]
  
  After --> Verify[Verify: Recompute using same formula]
  
  Verify --> NewRatios[New log ratios:<br/>yw: -1.715 − -1.609 = -0.106<br/>yl: -2.526 − -2.303 = -0.223]
  
  NewRatios --> NewDelta[New Δ = -0.106 − -0.223 = +0.118<br/>Now POSITIVE = prefers correct answer!]
  
  NewDelta --> NewSig[New σ of +0.118 ≈ 0.529<br/>52.9% confidence - much better!]
  
  NewSig --> NewLoss[New L = −log of 0.529 = 0.635]
  
  NewLoss --> Result([Training Success!<br/>Loss: 1.946 → 0.635<br/>Δ: -1.793 → +0.118<br/>Model now prefers correct answer 4])
  
  style Start fill:#555,stroke:#333,stroke-width:2px,color:#fff
  style Input fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Formula fill:#444,stroke:#333,stroke-width:2px,color:#fff
  style Step1 fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Ref fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Pol fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Step2 fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Ratios fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Step3 fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Delta fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Step4 fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Sigmoid fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Step5 fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Loss fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Step6 fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Update fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style After fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Verify fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style NewRatios fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style NewDelta fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style NewSig fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style NewLoss fill:#666,stroke:#333,stroke-width:1px,color:#fff
  style Result fill:#555,stroke:#333,stroke-width:3px,color:#fff