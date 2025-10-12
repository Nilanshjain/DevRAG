@echo off
echo ========================================
echo Starting Redis for DevRAG
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)

echo Docker is running...
echo.

REM Check if Redis container already exists
docker ps -a --filter "name=devrag-redis" --format "{{.Names}}" | findstr "devrag-redis" >nul 2>&1
if not errorlevel 1 (
    echo Redis container already exists. Starting it...
    docker start devrag-redis
) else (
    echo Creating new Redis container...
    docker run -d ^
        --name devrag-redis ^
        -p 6379:6379 ^
        redis:latest
)

echo.
echo ========================================
echo Redis is now running on localhost:6379
echo ========================================
echo.
echo To stop Redis: docker stop devrag-redis
echo To view logs: docker logs devrag-redis
echo.
pause
