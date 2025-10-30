"""
Wrapper entrypoint so `streamlit run app.py` works by calling the existing
`main()` in `streamlit_app.py`.
"""

from streamlit_app import main

if __name__ == '__main__':
    main()
