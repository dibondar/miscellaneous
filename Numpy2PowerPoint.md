# Importing Numpy arrays as curves in Microsoft PowerPoint

Assume you have two numpy arrays `x` and `y` that you want to plot as a curve in PowerPoint
```python
import numpy as np

x = np.linspace(-1, 1, 10)
y = x ** 2
```
Then, follow the following steps
1. In Python, using the function
```python
import numpy as np
import pyperclip
import os


def get_vba_code2plot(x: np.ndarray, y: np.ndarray, ppoint_slide_width=1280, ppoint_slide_height=360, temp_file="data.csv"):
    """
    Generate the VBA code and copy it into the clipboard to plot a curve specified by x and y
    :param x: 1D numpy.array
    :param y: 1D numpy.array
    :param ppoint_slide_width: (int) the width of the curve in pixes
    :param ppoint_slide_height: (int) the height of the curve in pixes
    :param temp_file: (str) file name for storing the 
    :return: None
    """
    # Pre processing arrays to fill the slide width
    x = x - x.min() 
    x = x * ppoint_slide_width / x.max()
    x = x.astype(np.int)

    y = y - y.min()
    y = y * ppoint_slide_height / y.max()
    y = y.astype(np.int)
    y = ppoint_slide_height - y

    # Saving post-processed data into the file
    np.savetxt(temp_file, np.vstack([x, y]).T, fmt='%d')
    
    # generate VBA code
    code = """
    Sub curve()

    Dim pts(1 to {size}, 1 to 2) as Single
    Dim intCount as Integer
    Dim iFileNum as Integer
    Dim arrSplitStrings As Variant
    Dim sBuf As String

    iFileNum = FreeFile()
    Open "{file_name}" For Input As iFileNum

    For intCount = 1 To {size}
        Line Input #iFileNum, sBuf
    
        arrSplitStrings = Split(sBuf)
        pts(intCount, 1) = Val(arrSplitStrings(0))
        pts(intCount, 2) = Val(arrSplitStrings(1))
    Next intCount

    Set myDocument = ActivePresentation.Slides(1)
    myDocument.Shapes.AddCurve SafeArrayOfPoints:=pts

    End Sub
    """.format(
        size=x.size,
        file_name=os.path.abspath(temp_file)
    )
    
    # copy the generated code to clipboard (optional)
    pyperclip.copy(code)
``` 
run
```python
get_vba_code2plot(x, y)
```
If the execution was successful, no message should appear. At this point the clipboard contain a generated Visual Basic for Application macros.

2. Open a PowerPoint presentation and go to menu `Tools\Macro\Visual Basic Editor`
3. In the newly open widow of Visual Basic Editor, right hand click on `VBAProject` and select `Insert\Module`
4. Paste the content of the clipboard (e.g., `Ctrl + V`) to the opened tex editor
5. Execute the macros. Note that you may be asked to grant an access to the temporary file specified in the argument `temp_file` of python function `get_vba_code2plot` (the default file name is `data.csv`). Also PowerPoint may throw an error if you want to plot too many points; in such a case, in step 3 above, call `get_vba_code2plot(x[::2], y[::2])` to slice your arrays.  

vualá  