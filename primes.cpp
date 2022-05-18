#include <bits/stdc++.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

int main(){
    FILE *fp = freopen("file.txt", "w+", stdout);
    vector <int> v;
    set <int> s;
    int n; cin>>n;
    auto start = high_resolution_clock::now();
    bool prime[n + 1];
	memset(prime, true, sizeof(prime));
	for (int p = 2; p * p <= n; p++) {
		if (prime[p] == true) {
			for (int i = p * p; i <= n; i += p) prime[i] = false;
		}
	}

	for (int p = 2; p <= n; p++)
		if (prime[p]) {
            v.push_back(p);
            s.insert(p);
        }
    vector <map<int,int>> factors(n+1);
    for (int i=2;i<=n;i++){
        factors[i]=factors[i-1];
        int p=i;
        for(auto it=s.begin(); it!=s.end();it++){
            int h=*it;
            if(h>i) break;
            if(p%h==0){
                while(p%h==0){
                    p/=h;
                    factors[i][h]++;
                }
            }
        }
        cout<<i<<": "<<endl;
        auto it=factors[i].begin();
        auto ip=factors[i].end();
        ip--;
        while(it!=ip){
            cout<<it->first<<"^"<<it->second<<" x ";
            it++;
        }
        cout<<it->first<<"^"<<it->second;
        cout<<endl<<endl;
    }
    fclose(fp);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << endl;
}

