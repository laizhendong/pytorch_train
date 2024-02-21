#include "GetXmlBoxes.h"

using namespace std;

int main(void)
{
	GetPicFromXmlBox myTest;
	myTest.XmlDir(R"(E:\DTYoriented_QAdata\Augmnet-illum\img-xml)");
	myTest.TxtSaveDir(R"(E:\DTYoriented_QAdata\Augmnet-illum\txt)");
	myTest.BoxLabel({ "__background__", "longjump", "crossjump" });
	myTest.AutoRunning();

	getchar();
}
