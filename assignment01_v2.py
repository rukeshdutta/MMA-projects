import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.linear_model import LinearRegression

class BidOptimizer:
    def __init__(self, conversion_rate=0.013, profit_per_conversion=100, daily_budget=10):
        """
        Initialize the bid optimizer with business parameters.
        
        Args:
            conversion_rate (float): Expected conversion rate (default: 1.3%)
            profit_per_conversion (float): Profit per conversion in dollars (default: $100)
            daily_budget (float): Maximum daily budget in dollars (default: $10)
        """
        self.conversion_rate = conversion_rate
        self.profit_per_conversion = profit_per_conversion
        self.daily_budget = daily_budget
        self.keywords = ['rent wheelchair', 'rental wheelchair', 'wheelchair rental']
        
        # Initialize data storage
        self.data = self._initialize_data()
        self.bounds = self._calculate_bounds()
        self.models = self._train_models()

    def _initialize_data(self):
        """Initialize keyword bid, clicks, and CPC data."""
        return {
            'rent wheelchair': pd.DataFrame({
                'bid': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
                'clicks': [1.00, 1.00, 1.50, 1.50, 1.50, 1.50, 1.50, 2.00, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50],
                'cpc': [0.21, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.63, 0.66, 0.70, 0.74, 0.77, 0.82]
            }),
            'rental wheelchair': pd.DataFrame({
                'bid': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                'clicks': [6.00, 7.00, 8.50, 9.00, 10.00, 11.50, 12.50, 12.50, 13.50, 14.50, 14.50, 15.50, 15.50, 16.50, 16.50],
                'cpc': [0.23, 0.26, 0.30, 0.35, 0.38, 0.43, 0.47, 0.52, 0.56, 0.61, 0.65, 0.69, 0.73, 0.77, 0.80]
            }),
            'wheelchair rental': pd.DataFrame({
                'bid': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
                'clicks': [6.00, 7.00, 8.50, 9.00, 10.00, 11.50, 12.50, 12.50, 13.50, 14.50, 14.50, 15.50, 15.50, 16.50, 16.50, 17.50],
                'cpc': [0.23, 0.26, 0.30, 0.35, 0.38, 0.43, 0.47, 0.52, 0.56, 0.61, 0.65, 0.69, 0.73, 0.77, 0.80, 0.85]
            })
        }

    def _calculate_bounds(self):
        """Calculate bid bounds for each keyword."""
        bounds = [(self.data[kw]['bid'].min(), self.data[kw]['bid'].max()) for kw in self.keywords]
        return bounds

    def _train_models(self):
        """Train regression models for clicks and CPC prediction."""
        models = {}
        for kw in self.keywords:
            X_bid = self.data[kw]['bid'].values.reshape(-1, 1)
            y_clicks = self.data[kw]['clicks'].values
            y_cpc = self.data[kw]['cpc'].values
            clicks_model = LinearRegression().fit(X_bid, y_clicks)
            cpc_model = LinearRegression().fit(X_bid, y_cpc)
            models[kw] = {'clicks_model': clicks_model, 'cpc_model': cpc_model}
        return models

    def calculate_cost(self, bids):
        """Calculate total cost for given bids without affecting optimization."""
        total_cost = 0
        for i, kw in enumerate(self.keywords):
            bid = bids[i]
            predicted_clicks = self.models[kw]['clicks_model'].predict([[bid]])[0]
            predicted_cpc = self.models[kw]['cpc_model'].predict([[bid]])[0]
            total_cost += predicted_clicks * predicted_cpc
        return total_cost

    def predict_daily_profit(self, bids, enforce_budget=True):
        """
        Calculate predicted daily profit for given bids.
        
        Args:
            bids (list): List of bid values for each keyword
            enforce_budget (bool): Whether to enforce the daily budget constraint
            
        Returns:
            float: Negative of predicted daily profit (for minimization)
        """
        total_profit = 0
        for i, kw in enumerate(self.keywords):
            bid = bids[i]
            predicted_clicks = self.models[kw]['clicks_model'].predict([[bid]])[0]
            predicted_cpc = self.models[kw]['cpc_model'].predict([[bid]])[0]
            daily_conversions = predicted_clicks * self.conversion_rate
            daily_revenue = daily_conversions * self.profit_per_conversion
            daily_cost = predicted_clicks * predicted_cpc
            
            total_profit += (daily_revenue - daily_cost)
    
        return -total_profit  # Return negative profit directly for minimization

    def optimize_bids(self, initial_bids=None, enforce_budget=True):
        """
        Optimize bids for all keywords.
        
        Args:
            initial_bids (list, optional): Initial bid values to start optimization
            enforce_budget (bool): Whether to enforce the daily budget constraint
            
        Returns:
            dict: Optimization results including optimal bids and expected metrics
        """
        if initial_bids is None:
            initial_bids = [0.5] * len(self.keywords)
        
        # Define budget constraint
        constraints = []
        if enforce_budget:
            constraints = [{'type': 'ineq', 'fun': lambda bids: self.daily_budget - self.calculate_cost(bids)}]

        result = minimize(
            self.predict_daily_profit, 
            initial_bids, 
            args=(enforce_budget,),
            bounds=self.bounds, 
            method='SLSQP',  # Switch to SLSQP to support constraints
            constraints=constraints
        )
        
        return self._format_results(result.x)

    def _format_results(self, optimal_bids):
        """Format optimization results into a readable dictionary."""
        results = {}
        total_metrics = {
            'total_cost': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_revenue': 0,
            'total_profit': 0
        }

        for kw, bid in zip(self.keywords, optimal_bids):
            clicks = self.models[kw]['clicks_model'].predict([[bid]])[0]
            cpc = self.models[kw]['cpc_model'].predict([[bid]])[0]
            daily_cost = clicks * cpc
            daily_conversions = clicks * self.conversion_rate
            daily_revenue = daily_conversions * self.profit_per_conversion
            daily_profit = daily_revenue - daily_cost
            
            results[kw] = {
                'optimal_bid': bid,
                'expected_clicks': clicks,
                'expected_cpc': cpc,
                'expected_daily_cost': daily_cost,
                'expected_daily_conversions': daily_conversions,
                'expected_daily_revenue': daily_revenue,
                'expected_daily_profit': daily_profit
            }
            
            # Update totals
            total_metrics['total_cost'] += daily_cost
            total_metrics['total_clicks'] += clicks
            total_metrics['total_conversions'] += daily_conversions
            total_metrics['total_revenue'] += daily_revenue
            total_metrics['total_profit'] += daily_profit
            
        results['campaign_totals'] = total_metrics
        return results

    def analyze_data_quality(self):
        """Analyze data quality for each keyword's bid dataset."""
        analysis = {}
        for kw in self.keywords:
            bids = self.data[kw]['bid'].values
            bid_diffs = np.diff(bids)
            
            analysis[kw] = {
                'data_points': len(self.data[kw]),
                'bid_range': {
                    'min': bids.min(),
                    'max': bids.max()
                },
                'bid_gaps': {
                    'average': bid_diffs.mean(),
                    'maximum': bid_diffs.max()
                }
            }
        return analysis

def main():
    # Initialize optimizer with default parameters
    optimizer = BidOptimizer()
    
    # Analyze data quality
    print("Analyzing data quality...")
    analysis = optimizer.analyze_data_quality()
    for kw, stats in analysis.items():
        print(f"\n{kw}:")
        print(f"Number of data points: {stats['data_points']}")
        print(f"Bid range: ${stats['bid_range']['min']:.2f} - ${stats['bid_range']['max']:.2f}")
        print(f"Average gap between bids: ${stats['bid_gaps']['average']:.2f}")
        print(f"Maximum gap between bids: ${stats['bid_gaps']['maximum']:.2f}")
    
    # Optimize bids with budget constraint
    print("\nOptimizing with budget constraint...")
    results_with_budget = optimizer.optimize_bids(enforce_budget=True)
    print("\nOptimization Results with Budget Constraint:")
    for kw, metrics in results_with_budget.items():
        if kw != 'campaign_totals':
            print(f"\n{kw}:")
            print(f"Optimal Bid: ${metrics['optimal_bid']:.2f}")
            print(f"Expected Daily Clicks: {metrics['expected_clicks']:.1f}")
            print(f"Expected CPC: ${metrics['expected_cpc']:.2f}")
            print(f"Expected Daily Cost: ${metrics['expected_daily_cost']:.2f}")
            print(f"Expected Daily Profit: ${metrics['expected_daily_profit']:.2f}")
    
    print("\nCampaign Totals with Budget Constraint:")
    totals = results_with_budget['campaign_totals']
    print(f"Total Daily Cost: ${totals['total_cost']:.2f}")
    print(f"Total Daily Clicks: {totals['total_clicks']:.1f}")
    print(f"Total Daily Conversions: {totals['total_conversions']:.2f}")
    print(f"Total Daily Revenue: ${totals['total_revenue']:.2f}")
    print(f"Total Daily Profit: ${totals['total_profit']:.2f}")
    
    # Optimize bids without budget constraint
    print("\nOptimizing without budget constraint...")
    results_no_budget = optimizer.optimize_bids(enforce_budget=False)
    print("\nOptimization Results without Budget Constraint:")
    for kw, metrics in results_no_budget.items():
        if kw != 'campaign_totals':
            print(f"\n{kw}:")
            print(f"Optimal Bid: ${metrics['optimal_bid']:.2f}")
            print(f"Expected Daily Clicks: {metrics['expected_clicks']:.1f}")
            print(f"Expected CPC: ${metrics['expected_cpc']:.2f}")
            print(f"Expected Daily Cost: ${metrics['expected_daily_cost']:.2f}")
            print(f"Expected Daily Profit: ${metrics['expected_daily_profit']:.2f}")
    
    print("\nCampaign Totals without Budget Constraint:")
    totals_no_budget = results_no_budget['campaign_totals']
    print(f"Total Daily Cost: ${totals_no_budget['total_cost']:.2f}")
    print(f"Total Daily Clicks: {totals_no_budget['total_clicks']:.1f}")
    print(f"Total Daily Conversions: {totals_no_budget['total_conversions']:.2f}")
    print(f"Total Daily Revenue: ${totals_no_budget['total_revenue']:.2f}")
    print(f"Total Daily Profit: ${totals_no_budget['total_profit']:.2f}")

if __name__ == "__main__":
    main()
