# Bid_genie_app
This is the construction bid price optimizer app (metro track projects), built as a part of my M.Tech thesis. This used a stacked ensemble to make predictions regarding the cost estimate and win probability at a markup, and using Monte Carlo simulation, it suggests the optimal markup and does the sensitivity analysis finally, the ML predictions are explained using SHAP


How to Use Bid Genie:

To explore the optimization and explainability features of this project, follow these steps:

1. Access the App: Click on the Live App link provided in the description or scan the QR code

2. Upload Data: Click the Browse files button in the sidebar to load project details from your historical dataset or enter the details manually.

3. AI Cost Estimation: If you're unsure of the cost, click the "Estimate Cost with AI" button to get a predicted cost based on project technical details.

4. Run Analysis: Click the "Analyze Bid" button. The system will:

      a. Identify the Optimal Markup for maximum expected profit.

      b. Simulate Risk of Loss using 1,000 Monte Carlo iterations.

      c. Generate a SHAP Waterfall Plot to explain which features contributed most to the win probability.

5. Download Report: Once the analysis is complete, you can download a full CSV report of the strategy simulation.

Do you know ?
In bidding auctions, the "winner" is often the person who most underestimated the costs. Therefore, winning can actually mean losing money.
