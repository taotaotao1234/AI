#include "http_updata.h"

char httpAddressPort[] ="http://127.0.0.1:80/zdinnerfast/servlet/anonymous/json/picCompareApi";
extern char param3[128] ;
extern char choice_src_addr[128] ; 
// 下发给本地HTTP
void upload_cur_http(short data)
{
    //char *POST_url = NULL;
    char POST_url[100] = {0};
    CURL *pCurl = NULL;
    struct curl_slist *headerlist = NULL;
    struct curl_httppost *post = NULL;
    struct curl_httppost *last = NULL;
    CURLcode res;
    res = curl_global_init(CURL_GLOBAL_ALL);
    if(res != CURLE_OK)
    {
        printf("curl_global_init failed!\n");
        return;
    }
    
    char param_data[128] = {0};
    sprintf(param_data,"%s",param3);

    sprintf(POST_url,"%s",httpAddressPort);    // 内网接口
    curl_formadd(&post, &last,CURLFORM_COPYNAME, "id",              			
        CURLFORM_COPYCONTENTS, param_data,                                       						
        CURLFORM_END);

    char msg_data[64] = {0};
    sprintf(msg_data,"%d",data);
    
    sprintf(POST_url,"%s",httpAddressPort);    // 内网接口
    curl_formadd(&post, &last,CURLFORM_COPYNAME, "howManyDiff",              			
        CURLFORM_COPYCONTENTS, msg_data,                                       						
        CURLFORM_END);
    
    char src_pictrue_data[128] = {0};
    sprintf(src_pictrue_data,"%s",choice_src_addr);
    
    sprintf(POST_url,"%s",httpAddressPort);    // 内网接口
    curl_formadd(&post, &last,CURLFORM_COPYNAME, "basePath",              			
        CURLFORM_COPYCONTENTS, src_pictrue_data,                                       						
        CURLFORM_END);

    pCurl = curl_easy_init();
    if (NULL == pCurl)
    {
        printf("\rpCurl initialization failed!\n");
        return;
    }
    curl_easy_setopt(pCurl, CURLOPT_URL, POST_url);					
    curl_easy_setopt(pCurl, CURLOPT_HTTPPOST, post);	
    curl_easy_setopt(pCurl, CURLOPT_TIMEOUT,15);
 
    curl_easy_setopt(pCurl, CURLOPT_TCP_KEEPALIVE, 1L);
	
    res = curl_easy_perform(pCurl);					
    if (res == CURLE_OK)					
    {			
        printf("id=%s,Different number of picture = [%s]\n", param_data, msg_data);			   					  
    }
    else
    {
        printf("id=%s,Different number of pictures = [%s]\n", param_data, msg_data);
        printf("curl_easy_perform() failed，error code is:%s\n", curl_easy_strerror(res));
    }					
    curl_easy_cleanup(pCurl);
}


