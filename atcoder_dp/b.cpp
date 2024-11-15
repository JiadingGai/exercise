#include <bits/stdc++.h>
using namespace std;

int main() {
    auto x = std::vector({1,2,3});
    x.push_back(4);
    ios::sync_with_stdio(false);
    cin.tie(0);
    int n, k;
    cin >> n;
    cin >> k;
    vector<int> h(n);
    for (int &x : h) {
        cin >> x;
    }
    vector<int> dp(n, 0x7fffffff);
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j <= i + k; j++) {
        if (j < n)
          dp[j] = min(dp[j], dp[i] + abs(h[j] - h[i]));
      }
    }
    cout << dp[n-1] <<"\n";
    return 0;
}
