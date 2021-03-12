#include<bits/stdc++.h>
using namespace std;
char s[10012];
int vis[123123];
int main() {
    freopen("mix.txt","r",stdin);
    freopen("data.txt","w",stdout);
    int id,sentence;
    while(~scanf("%d",&id)) {
        memset(s,0,sizeof s);
        scanf("%d",&sentence);
        gets(s+1);
        //if(vis[sentence]) continue;
        vis[sentence]=1;
        int len=strlen(s+1);
        while(len>=2&&(s[len]=='\t'||s[len]=='\n'||s[len]=='\r'||s[len]==' '||isdigit(s[len]))) len--;
        s[len+1]='\0';
        printf("%s\n",s+2);
    }
}
