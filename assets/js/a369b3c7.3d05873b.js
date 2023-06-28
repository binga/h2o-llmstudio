"use strict";(self.webpackChunksite=self.webpackChunksite||[]).push([[617],{3905:(t,e,a)=>{a.d(e,{Zo:()=>p,kt:()=>f});var n=a(7294);function r(t,e,a){return e in t?Object.defineProperty(t,e,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[e]=a,t}function o(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),a.push.apply(a,n)}return a}function i(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?o(Object(a),!0).forEach((function(e){r(t,e,a[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):o(Object(a)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))}))}return t}function l(t,e){if(null==t)return{};var a,n,r=function(t,e){if(null==t)return{};var a,n,r={},o=Object.keys(t);for(n=0;n<o.length;n++)a=o[n],e.indexOf(a)>=0||(r[a]=t[a]);return r}(t,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(t);for(n=0;n<o.length;n++)a=o[n],e.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(t,a)&&(r[a]=t[a])}return r}var c=n.createContext({}),s=function(t){var e=n.useContext(c),a=e;return t&&(a="function"==typeof t?t(e):i(i({},e),t)),a},p=function(t){var e=s(t.components);return n.createElement(c.Provider,{value:e},t.children)},d="mdxType",u={inlineCode:"code",wrapper:function(t){var e=t.children;return n.createElement(n.Fragment,{},e)}},m=n.forwardRef((function(t,e){var a=t.components,r=t.mdxType,o=t.originalType,c=t.parentName,p=l(t,["components","mdxType","originalType","parentName"]),d=s(a),m=r,f=d["".concat(c,".").concat(m)]||d[m]||u[m]||o;return a?n.createElement(f,i(i({ref:e},p),{},{components:a})):n.createElement(f,i({ref:e},p))}));function f(t,e){var a=arguments,r=e&&e.mdxType;if("string"==typeof t||r){var o=a.length,i=new Array(o);i[0]=m;var l={};for(var c in e)hasOwnProperty.call(e,c)&&(l[c]=e[c]);l.originalType=t,l[d]="string"==typeof t?t:r,i[1]=l;for(var s=2;s<o;s++)i[s]=a[s];return n.createElement.apply(null,i)}return n.createElement.apply(null,a)}m.displayName="MDXCreateElement"},1904:(t,e,a)=>{a.r(e),a.d(e,{assets:()=>c,contentTitle:()=>i,default:()=>u,frontMatter:()=>o,metadata:()=>l,toc:()=>s});var n=a(7462),r=(a(7294),a(3905));const o={},i="Supported data connectors and format",l={unversionedId:"guide/datasets/data-connectors-format",id:"guide/datasets/data-connectors-format",title:"Supported data connectors and format",description:"Data connectors",source:"@site/docs/guide/datasets/data-connectors-format.md",sourceDirName:"guide/datasets",slug:"/guide/datasets/data-connectors-format",permalink:"/h2o-llmstudio/guide/datasets/data-connectors-format",draft:!1,tags:[],version:"current",frontMatter:{},sidebar:"defaultSidebar",previous:{title:"Concepts",permalink:"/h2o-llmstudio/concepts"},next:{title:"Import a dataset",permalink:"/h2o-llmstudio/guide/datasets/import-dataset"}},c={},s=[{value:"Data connectors",id:"data-connectors",level:2},{value:"Data format",id:"data-format",level:2}],p={toc:s},d="wrapper";function u(t){let{components:e,...a}=t;return(0,r.kt)(d,(0,n.Z)({},p,a,{components:e,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"supported-data-connectors-and-format"},"Supported data connectors and format"),(0,r.kt)("h2",{id:"data-connectors"},"Data connectors"),(0,r.kt)("p",null,"H2O LLM Studio supports the following data connectors to access or upload external data sources."),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Upload"),": Upload a local dataset from your machine. "),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Local"),": Specify the file location of the dataset on your machine. "),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"AWS S3 (Amazon AWS S3)"),": Connect to an Amazon AWS S3 data bucket. "),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"Kaggle"),": Connect to a Kaggle dataset. ")),(0,r.kt)("h2",{id:"data-format"},"Data format"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"Each data connector requires either a single ",(0,r.kt)("inlineCode",{parentName:"p"},".csv")," or ",(0,r.kt)("inlineCode",{parentName:"p"},".pq")," file, or the data to be in a ",(0,r.kt)("inlineCode",{parentName:"p"},".zip")," file for a successful import. ")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"H2O LLM studio requires a ",(0,r.kt)("inlineCode",{parentName:"p"},".csv")," file with a minimum of two columns, where one contains the instructions and the other has the model\u2019s expected output. You can also include an additional validation dataframe in the same format or allow for an automatic train/validation split to assess the model\u2019s performance.")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"Optionally, a ",(0,r.kt)("strong",{parentName:"p"},"Parent Id")," can be used for training nested data prompts that are linked to a parent question."))))}u.isMDXComponent=!0}}]);